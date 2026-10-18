# -*- coding: utf-8 -*-
"""
NSD fMRI -> LLM caption embedding decoding pipeline (Doerig et al. 2025 style).

Pipeline:
  1. Load per-session betas (int16, -32768 = missing) restricted to the
     'streams' visual ROI.
  2. Map each trial to its image via responses.tsv (73KID = nsdId + 1),
     then to the precomputed MPNet caption embedding Y.
  3. Split by IMAGE (no image in both train and test) to avoid repeat leakage.
  4. Ridge regression  beta(voxels) -> embedding(768)  with lambda chosen by
     CV on the training trials.
  5. Evaluate: Pearson correlation between predicted and target embeddings,
     plus a 5-caption consistency noise ceiling.

Usage:
  python decode_nsd.py [--max-sessions 2] [--max-voxels 5000] [--test-images 500]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
NSD = ROOT / "data" / "nsd"
BETAS = NSD / "nsddata_betas" / "ppdata" / "subj01" / "func1pt8mm" / "betas_fithrf_GLMdenoise_RR"
OUT = ROOT / "outputs"
MISSING = -32768


def load_trial_table():
    """rows = [(session, 73KID), ...] in chronological order per session."""
    rows = []
    with open(NSD / "nsddata" / "ppdata" / "subj01" / "behav" / "responses.tsv",
              encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append((int(r["SESSION"]), int(r["73KID"])))
    return rows


def load_streams_voxels(max_voxels, r2_weight=True):
    """Voxel indices inside streams ROI; optionally keep the most reliable ones."""
    import nibabel as nib
    roi = np.asarray(nib.load(NSD / "nsddata" / "ppdata" / "subj01" / "func1pt8mm" /
                              "roi" / "streams.nii.gz").dataobj)
    mask = roi > 0                                   # 7 sub-ROIs, 0 = outside
    idx = np.where(mask.ravel())[0]
    if max_voxels and len(idx) > max_voxels:
        if r2_weight:
            r2 = np.asarray(nib.load(NSD / "nsddata" / "ppdata" / "subj01" / "func1pt8mm" /
                                     "R2.nii.gz").dataobj).ravel()[idx]
            idx = idx[np.argsort(-r2)[:max_voxels]]
        else:
            idx = idx[:max_voxels]
    idx = np.sort(idx)
    return idx, len(idx)


def load_betas_sessions(sessions, voxel_idx):
    """Stack trials of the given sessions -> (n_trials, n_voxels) float32."""
    import nibabel as nib
    X, sess_of = [], []
    for s in sessions:
        img = nib.load(BETAS / f"betas_session{s:02d}.nii.gz")
        assert img.shape[3] == 750, f"unexpected trial count in session {s}"
        arr = np.asarray(img.dataobj)                # int16, ~1 GB per session
        sel = arr.reshape(-1, img.shape[3])[voxel_idx].T   # (750, n_sel)
        X.append(sel.astype(np.float32))
        sess_of.extend([s] * 750)
        del arr
        print(f"  session {s:02d} loaded", flush=True)
    return np.vstack(X), np.asarray(sess_of)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-sessions", type=int, default=40)
    ap.add_argument("--max-voxels", type=int, default=5000)
    ap.add_argument("--test-images", type=int, default=500)
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0])
    args = ap.parse_args()

    # ---- assemble trials ------------------------------------------------
    print("loading trial table...")
    rows = load_trial_table()
    n_sessions = min(args.max_sessions, 40)
    # keep only trials of sessions that exist on disk
    available = [s for s in range(1, n_sessions + 1)
                 if (BETAS / f"betas_session{s:02d}.nii.gz").exists()]
    print(f"sessions available: {available[:5]}... ({len(available)} total)")
    keep = np.asarray([s in available for s, _ in rows])
    rows_s = [r for r, k in zip(rows, keep) if k]

    print("loading streams ROI...")
    voxel_idx, n_vox = load_streams_voxels(args.max_voxels)
    print(f"using {n_vox} voxels")

    print("loading betas...")
    X, sess_of = load_betas_sessions(available, voxel_idx)
    n_trials = X.shape[0]
    assert n_trials == len(rows_s)

    # missing voxels: -32768 markers -> 0 (they sit outside the brain)
    X[X == MISSING] = 0.0

    # map trials -> Y rows: 73KID = nsdId + 1
    z = np.load(OUT / "nsd_subj01_caption_embeddings.npz", allow_pickle=True)
    Y_all, nsd_ids = z["Y"], z["nsdId"]
    nsd_to_row = {int(n): i for i, n in enumerate(nsd_ids)}
    y_rows = np.asarray([nsd_to_row[k - 1] for _, k in rows_s])
    Y = Y_all[y_rows]
    print(f"X {X.shape}, Y {Y.shape}")

    # ---- split by image (no leakage across repeats) ----------------------
    uniq_img = np.unique(y_rows)
    rng = np.random.default_rng(0)
    test_imgs = set(rng.choice(uniq_img, size=min(args.test_images, len(uniq_img)),
                               replace=False).tolist())
    is_test = np.asarray([r in test_imgs for r in y_rows])
    print(f"train trials {int((~is_test).sum())}, test trials {int(is_test.sum())}")

    # ---- ridge with CV on lambda ----------------------------------------
    Xtr, Ytr = X[~is_test], Y[~is_test]
    Xte, Yte = X[is_test], Y[is_test]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_z = (Xtr - mu) / sd
    Xte_z = (Xte - mu) / sd

    # 3-fold CV to pick lambda (Gram computed once per fold, reused across lambdas)
    n = Xtr_z.shape[0]
    folds = np.array_split(rng.permutation(n), 3)
    fold_cache = []                                   # (A_tr, B_tr, va_indices)
    for f in range(3):
        va = folds[f]
        tr = np.setdiff1d(np.arange(n), va)
        A_tr = Xtr_z[tr].T @ Xtr_z[tr]
        B_tr = Xtr_z[tr].T @ Ytr[tr]
        fold_cache.append((A_tr, B_tr, va, tr))
        print(f"  fold {f+1}/3 Gram done", flush=True)
    best_l, best_cv = None, -np.inf
    for lam in args.lambdas:
        cors = []
        for A_tr, B_tr, va, tr in fold_cache:
            A = A_tr + lam * len(tr) * np.eye(n_vox, dtype=np.float32)
            W = np.linalg.solve(A, B_tr)
            P = Xtr_z[va] @ W
            c = np.sum(P * Ytr[va], axis=1)          # cosine (both normalised)
            cors.append(float(np.mean(c)))
        m = float(np.mean(cors))
        print(f"  lambda={lam:<7} CV cosine={m:.4f}")
        if m > best_cv:
            best_cv, best_l = m, lam
    print(f"best lambda={best_l} (CV cosine {best_cv:.4f})")

    A = Xtr_z.T @ Xtr_z + best_l * n * np.eye(n_vox, dtype=np.float32)
    W = np.linalg.solve(A, Xtr_z.T @ Ytr)
    pred = Xte_z @ W

    # ---- evaluation ------------------------------------------------------
    cos = np.sum(pred * Yte, axis=1)
    cos_all = pred @ Y_all.T                           # (n_test, 10000)
    rank = (cos_all < cos[:, None]).sum(1)             # rank of true target
    print("\n===== decoding performance (test trials) =====")
    print(f"cosine(pred, target): mean {cos.mean():.4f}  median {np.median(cos):.4f}")
    print(f"mean rank of true image among 10,000: {rank.mean():.1f} "
          f"(chance ~5000) | top-1: {(rank == 0).mean():.3f} "
          f"top-1%: {(rank < 100).mean():.3f}")

    # noise ceiling: 5-caption internal consistency (paper's definition)
    caps = z["n_caps"]
    print(f"(noise ceiling uses the 5-caption consistency of the held-out images)")

    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / "decode_pred_test.npz",
                        pred=pred, target=Yte, cos=cos, rank=rank,
                        y_rows=y_rows[is_test], best_lambda=best_l,
                        n_vox=n_vox, sessions=available)
    print("saved -> outputs/decode_pred_test.npz")


if __name__ == "__main__":
    main()
