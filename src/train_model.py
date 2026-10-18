# -*- coding: utf-8 -*-
"""
Train an fMRI->embedding ridge decoder and save a self-contained model bundle
for the interactive decoding app (decode_app.py).

The bundle contains: weights W, voxel normalisation stats, the test-split
trials' beta vectors (pre-extracted for instant loading in the app), each
test image's ground-truth captions, and provenance (session, timepoint).

Usage:
    python train_model.py [--max-voxels 10000] [--test-images 800] [--tag s09]
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
NSD = ROOT / "data" / "nsd"
BETAS = NSD / "nsddata_betas" / "ppdata" / "subj01" / "func1pt8mm" / "betas_fithrf_GLMdenoise_RR"
OUT = ROOT / "outputs" / "ridge_models"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decode_nsd import (MISSING, load_betas_sessions, load_streams_voxels,
                        load_trial_table)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-voxels", type=int, default=10000)
    ap.add_argument("--test-images", type=int, default=800)
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    args = ap.parse_args()

    t0 = time.time()
    # ---- data ------------------------------------------------------------
    rows = load_trial_table()
    available = [s for s in range(1, 41)
                 if (BETAS / f"betas_session{s:02d}.nii.gz").exists()]
    keep = np.asarray([s in available for s, _ in rows])
    rows_s = [r for r, k in zip(rows, keep) if k]
    print(f"sessions used: {len(available)} -> {len(rows_s)} trials", flush=True)

    # provenance: (session, timepoint) for every kept trial, in load order
    tp_of, sess_of_rows = [], []
    counters = {s: 0 for s in available}
    for s, _ in rows_s:
        tp_of.append(counters[s]); sess_of_rows.append(s); counters[s] += 1
    assert all(v == 750 for v in counters.values()), "session trial count != 750"

    voxel_idx, n_vox = load_streams_voxels(args.max_voxels)
    print(f"voxels: {n_vox}", flush=True)
    X, _ = load_betas_sessions(available, voxel_idx)
    X[X == MISSING] = 0.0

    z = np.load(ROOT / "outputs" / "nsd_subj01_caption_embeddings.npz",
                allow_pickle=True)
    Y_all, nsd_ids, coco_ids = z["Y"], z["nsdId"], z["cocoId"]
    nsd_to_row = {int(n): i for i, n in enumerate(nsd_ids)}
    y_rows = np.asarray([nsd_to_row[k - 1] for _, k in rows_s])
    Y = Y_all[y_rows]
    print(f"X {X.shape}  Y {Y.shape}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- split by image ---------------------------------------------------
    uniq = np.unique(y_rows)
    rng = np.random.default_rng(0)
    test_imgs = set(rng.choice(uniq, size=min(args.test_images, len(uniq)),
                               replace=False).tolist())
    is_test = np.asarray([r in test_imgs for r in y_rows])
    Xtr, Ytr = X[~is_test], Y[~is_test]
    print(f"train trials {len(Xtr)} | test trials {int(is_test.sum())}", flush=True)

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_z = (Xtr - mu) / sd

    # ---- ridge with 3-fold CV ---------------------------------------------
    n = len(Xtr_z)
    folds = np.array_split(rng.permutation(n), 3)
    cache = []
    for f in range(3):
        va = folds[f]
        tr = np.setdiff1d(np.arange(n), va)
        cache.append((Xtr_z[tr].T @ Xtr_z[tr], Xtr_z[tr].T @ Ytr[tr], va, tr))
        print(f"fold {f+1}/3 Gram done ({time.time()-t0:.0f}s)", flush=True)
    best_l, best_cv = None, -np.inf
    for lam in args.lambdas:
        cors = []
        for A_tr, B_tr, va, tr in cache:
            A = A_tr + lam * len(tr) * np.eye(n_vox, dtype=np.float32)
            W = np.linalg.solve(A, B_tr)
            c = np.sum((Xtr_z[va] @ W) * Ytr[va], axis=1)
            cors.append(float(np.mean(c)))
        m = float(np.mean(cors))
        print(f"  lambda={lam:<6} CV cosine={m:.4f}", flush=True)
        if m > best_cv:
            best_cv, best_l = m, lam
    print(f"best lambda {best_l} (CV cosine {best_cv:.4f})", flush=True)

    A = Xtr_z.T @ Xtr_z + best_l * n * np.eye(n_vox, dtype=np.float32)
    W = np.linalg.solve(A, Xtr_z.T @ Ytr)
    pred = (X[is_test] - mu) / sd @ W
    cos = np.sum(pred * Y[is_test], axis=1)
    print(f"TEST cosine: mean {cos.mean():.4f} median {np.median(cos):.4f}",
          flush=True)

    # ---- save bundle -------------------------------------------------------
    te_idx = np.where(is_test)[0]
    te_rows = y_rows[te_idx]
    img_uniq, te_img = np.unique(te_rows, return_inverse=True)

    # ground-truth captions per test image, via cocoId
    import json
    coco_caps = {}
    for split in ["train", "val"]:
        d = json.loads((ROOT / "data" / "coco" / f"captions_{split}2017.json")
                       .read_text(encoding="utf-8"))
        for a in d["annotations"]:
            coco_caps.setdefault(a["image_id"], []).append(a["caption"].strip())

    def get5(coco):
        cs = coco_caps.get(coco, [])[:5]
        return cs + [""] * (5 - len(cs))

    img_caps = np.asarray([get5(int(coco_ids[r])) for r in img_uniq],
                          dtype=object)

    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"s{len(available):02d}_v{n_vox}_{time.strftime('%Y%m%d')}"
    path = OUT / f"ridge_{tag}.npz"
    np.savez_compressed(
        path,
        W=W.astype(np.float32), mu=mu.astype(np.float32), sd=sd.astype(np.float32),
        best_lambda=best_l, cv_cosine=best_cv,
        test_cos_mean=float(cos.mean()), test_cos_median=float(np.median(cos)),
        sessions=np.asarray(available),
        Xte=X[te_idx].astype(np.float32),          # test betas (n_te, n_vox)
        te_Y=Y[is_test].astype(np.float32),        # their target embeddings
        te_img=te_img,                             # image index per trial
        te_session=np.asarray(sess_of_rows)[te_idx],
        te_tp=np.asarray(tp_of)[te_idx],
        img_nsdId=nsd_ids[img_uniq], img_cocoId=coco_ids[img_uniq],
        img_caps=img_caps, n_vox=n_vox,
        train_date=time.strftime("%Y-%m-%d %H:%M"))
    print(f"saved model bundle -> {path}", flush=True)


if __name__ == "__main__":
    main()
