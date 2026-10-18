# -*- coding: utf-8 -*-
"""
Robust parallel downloader for NSD subj01 (public S3 bucket, no credentials).

- 16 parallel HTTP range connections per file (S3 per-connection throttling
  makes single-stream ~80 KB/s; parallelism gives ~0.9 MB/s).
- Resumable at file and chunk granularity: rerun the script anytime.
- Downloads in priority order: small essentials first, then betas sessions.

Usage:  python download_nsd.py [--workers 16] [--chunk-mb 8]
"""
import argparse
import os
import sys
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BUCKET = "https://natural-scenes-dataset.s3.amazonaws.com"
ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "nsd"
LOG = ROOT / "outputs" / "nsd_download.log"

PREFIXES_SMALL = [
    "nsddata/ppdata/subj01/behav/",
    "nsddata/experiments/nsd/",
    "nsddata/ppdata/subj01/func1pt8mm/",
]
PREFIXES_BIG = [
    "nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/",
]
SKIP_SUBSTR = ["screencapture", "nsdimagery", "nsdsynthetic", ".hdf5"]

_print_lock = threading.Lock()


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with _print_lock:
        print(line, flush=True)
        LOG.parent.mkdir(exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def list_prefix(session, prefix, retries=8):
    """Return [(key, size), ...] for all objects under prefix."""
    out, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        for attempt in range(1, retries + 1):
            try:
                r = session.get(BUCKET, params=params, timeout=60)
                r.raise_for_status()
                break
            except requests.RequestException:
                if attempt == retries:
                    raise
                time.sleep(min(2 ** attempt, 60))
        root = ET.fromstring(r.text)
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        for c in root.findall("s3:Contents", ns):
            out.append((c.find("s3:Key", ns).text, int(c.find("s3:Size", ns).text)))
        if root.find("s3:IsTruncated", ns).text != "true":
            return out
        token = root.find("s3:NextContinuationToken", ns).text


def fetch_range(session, url, start, end, path, retries=12):
    """Download one byte range to path, with retries. Returns bytes written."""
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=(30, 90))
            if r.status_code in (200, 206) and len(r.content) == end - start + 1:
                tmp = path.with_suffix(".tmp")
                tmp.write_bytes(r.content)
                os.replace(tmp, path)
                return len(r.content)
        except requests.RequestException:
            pass
        time.sleep(min(2 ** attempt, 60))
    raise RuntimeError(f"chunk failed after {retries} tries: {path.name}")


def download_file(session, key, size, workers, chunk):
    """Download one S3 key to DEST/<key>, chunked and parallel."""
    url = f"{BUCKET}/{key}"
    target = DEST / key
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == size:
        parts_dir = target.parent / (target.name + ".parts")  # orphan cleanup
        if parts_dir.exists():
            for p in parts_dir.iterdir():
                p.unlink()
            parts_dir.rmdir()
        return 0.0  # already complete

    t0 = time.time()
    if size <= chunk:
        fetch_range(session, url, 0, size - 1, target.with_suffix(".direct"))
        os.replace(target.with_suffix(".direct"), target)
        return time.time() - t0

    parts_dir = target.parent / (target.name + ".parts")
    parts_dir.mkdir(exist_ok=True)
    jobs = []
    for i, start in enumerate(range(0, size, chunk)):
        end = min(start + chunk, size) - 1
        ppath = parts_dir / f"{i:06d}"
        if ppath.exists() and ppath.stat().st_size == end - start + 1:
            continue  # chunk already done
        jobs.append((start, end, ppath))

    if jobs:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(fetch_range, session, url, s, e, p)
                    for s, e, p in jobs]
            for f in as_completed(futs):
                f.result()

    # assemble
    with open(target, "wb") as out:
        for i in range(0, (size + chunk - 1) // chunk):
            ppath = parts_dir / f"{i:06d}"
            with open(ppath, "rb") as pf:
                out.write(pf.read())
    if target.stat().st_size != size:
        target.unlink()
        raise RuntimeError(f"size mismatch after assembly: {key}")
    for p in parts_dir.iterdir():
        p.unlink()
    parts_dir.rmdir()
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk-mb", type=int, default=8)
    args = ap.parse_args()
    chunk = args.chunk_mb * 1024 * 1024

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32)
    session.mount("https://", adapter)

    manifest = []
    for prefix in PREFIXES_SMALL + PREFIXES_BIG:
        files = list_prefix(session, prefix)
        files = [(k, s) for k, s in files
                 if not any(x in k for x in SKIP_SUBSTR)]
        manifest.extend(files)

    # order: small prefixes first (already in order), then sessions sorted
    def sort_key(ks):
        k = ks[0]
        if "betas_session" not in k:
            return (0, k)
        return (1, k)

    manifest.sort(key=sort_key)
    total = sum(s for _, s in manifest)
    log(f"manifest: {len(manifest)} files, {total / 1e9:.2f} GB total")

    done_bytes = 0
    n_big_done = 0
    n_big = sum(1 for k, _ in manifest if "betas_session" in k)
    t_start = time.time()
    failures = []
    for i, (key, size) in enumerate(manifest):
        is_big = "betas_session" in key
        try:
            dt = download_file(session, key, size, args.workers, chunk)
        except Exception as e:
            log(f"file {i+1}/{len(manifest)}: {key.split('/')[-1]} FAILED ({e})")
            failures.append((key, size))
            continue
        done_bytes += size
        if is_big:
            n_big_done += 1
            rate = done_bytes / max(time.time() - t_start, 1) / 1e6
            remain = (total - done_bytes) / max(rate, 0.01) / 1e6 / 3600
            log(f"session {n_big_done}/{n_big}: {key.split('/')[-1]} "
                f"({size/1e9:.2f} GB in {dt/60:.1f} min) "
                f"| total {done_bytes/1e9:.1f}/{total/1e9:.1f} GB "
                f"| ~{rate:.2f} MB/s | ETA {remain:.1f} h")
        elif dt > 0 or size > 1e6:
            log(f"file {i+1}/{len(manifest)}: {key.split('/')[-1]} done")

    # retry failed files up to 3 more passes
    for attempt in range(1, 4):
        if not failures:
            break
        log(f"retry pass {attempt}: {len(failures)} failed file(s)")
        still = []
        for key, size in failures:
            try:
                download_file(session, key, size, args.workers, chunk)
                done_bytes += size
            except Exception as e:
                log(f"still failing: {key.split('/')[-1]} ({e})")
                still.append((key, size))
        failures = still
    if failures:
        log(f"GIVING UP on {len(failures)} file(s): "
            + ", ".join(k.split('/')[-1] for k, _ in failures))
        sys.exit(1)
    log("ALL DONE")


if __name__ == "__main__":
    main()
