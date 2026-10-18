# -*- coding: utf-8 -*-
"""Wait until all 40 NSD betas sessions are downloaded, then run the full decode."""
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
BETAS = ROOT / "data" / "nsd" / "nsddata_betas" / "ppdata" / "subj01" / "func1pt8mm" / "betas_fithrf_GLMdenoise_RR"
TOTAL = 40


def sessions_ready():
    return sum(1 for s in range(1, TOTAL + 1)
               if (BETAS / f"betas_session{s:02d}.nii.gz").exists())


if __name__ == "__main__":
    while True:
        n = sessions_ready()
        print(f"[{time.strftime('%H:%M:%S')}] sessions ready: {n}/{TOTAL}", flush=True)
        if n == TOTAL:
            # give the downloader 2 min to finish its last file bookkeeping
            time.sleep(120)
            print("all sessions ready -> launching full decode", flush=True)
            break
        time.sleep(600)
    cmd = [sys.executable, str(ROOT / "src" / "decode_nsd.py"),
           "--max-sessions", "40", "--max-voxels", "10000", "--test-images", "1000"]
    env = {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "PATH": ""}
    env = {**dict(__import__("os").environ), **{k: v for k, v in env.items() if v}}
    with open(ROOT / "outputs" / "decode_full_run.log", "w", encoding="utf-8") as logf:
        subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, check=True)
    print("full decode finished -> outputs/decode_full_run.log", flush=True)
