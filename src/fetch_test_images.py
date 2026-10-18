# -*- coding: utf-8 -*-
"""
Download only the test images needed by the decode app, by HTTP-Range
extracting them from the COCO zip mirror (no full 19GB download).

Usage:  python fetch_test_images.py [--workers 8]
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "data" / "coco" / "images"
MIRROR = "https://hf-mirror.com/datasets/pcuenq/coco-2017-mirror/resolve/main"


def needed_images():
    """(split, cocoId) for every test image in the newest model bundle."""
    bundles = sorted((ROOT / "outputs" / "ridge_models").glob("*.npz"))
    assert bundles, "no model bundle found"
    b = np.load(bundles[-1], allow_pickle=True)
    y = np.load(ROOT / "outputs" / "nsd_subj01_caption_embeddings.npz",
                allow_pickle=True)
    split_of = {int(n): s for n, s in zip(y["nsdId"], y["cocoSplit"])}
    out, missing_split = [], 0
    for nsd, coco in zip(b["img_nsdId"], b["img_cocoId"]):
        s = split_of.get(int(nsd))
        if s:
            out.append((s, int(coco)))
        else:
            missing_split += 1
    print(f"model: {bundles[-1].name} -> {len(out)} test images "
          f"({missing_split} without split info)")
    return out


def fetch(split, coco_id):
    """Extract one image from the remote zip. Returns (status, path)."""
    name = f"{coco_id:012d}.jpg"
    dest = IMG_DIR / split / name
    if dest.exists() and dest.stat().st_size > 0:
        return "skip", dest
    from remotezip import RemoteZip
    url = f"{MIRROR}/{split}.zip"
    for attempt in range(3):
        try:
            with RemoteZip(url) as z:
                z.extract(f"{split}/{name}", IMG_DIR)
            return "ok", dest
        except Exception as e:
            time.sleep(2 ** attempt)
    return "fail", dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    items = needed_images()
    for split in {s for s, _ in items}:
        (IMG_DIR / split).mkdir(parents=True, exist_ok=True)

    t0, done, fails = time.time(), 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch, s, c): (s, c) for s, c in items}
        for f in as_completed(futs):
            status, path = f.result()
            done += 1
            if status == "fail":
                fails.append(path)
            if done % 50 == 0 or done == len(items):
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(items)} ({rate:.1f} img/s, "
                      f"{len(fails)} failed)", flush=True)
    if fails:
        print("FAILED:", [p.name for p in fails[:10]])
        sys.exit(1)
    print("all test images ready under data/coco/images/")


if __name__ == "__main__":
    main()
