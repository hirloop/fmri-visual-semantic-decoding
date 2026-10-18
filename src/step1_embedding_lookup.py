# -*- coding: utf-8 -*-
"""
Step 1 toy demo: MPNet sentence embedding + dictionary nearest-neighbour lookup.

Mimics the decoding pipeline of Doerig et al. (2025, Nat. Mach. Intell.):

  target embedding  = mean MPNet embedding of the 5 human captions of an image
  predicted embedding = target + Gaussian noise   (simulates "decoded from fMRI")
  reconstruction    = nearest neighbour (cosine) among a held-out caption dictionary

The dictionary images and the query images are DISJOINT, so a hit cannot come
from looking up training captions.

Usage:
    python step1_embedding_lookup.py [--model PATH] [--n-dict 5000] [--n-query 1000]
"""
import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"


def load_captions():
    """Return {image_id: [caption, ...]} from the official COCO annotations,
    falling back to the sentence-transformers jsonl if the JSON is absent."""
    official = DATA / "coco" / "captions_train2017.json"
    if official.exists():
        d = json.loads(official.read_text(encoding="utf-8"))
        by_img = defaultdict(list)
        for a in d["annotations"]:
            by_img[a["image_id"]].append(a["caption"].strip())
        return dict(by_img)
    fallback = DATA / "gcc" / "coco_captions.jsonl.gz"
    if fallback.exists():
        by_img = {}
        with gzip.open(fallback, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                by_img[i] = [c.strip() for c in json.loads(line)["set"] if c.strip()]
        return by_img
    raise FileNotFoundError("No COCO caption source found under data/")


def embed_captions(model, texts, batch_size=128, desc="embed"):
    from tqdm import tqdm
    chunks = [
        model.encode(texts[i:i + batch_size], batch_size=batch_size,
                     normalize_embeddings=True, convert_to_numpy=True)
        for i in tqdm(range(0, len(texts), batch_size), desc=desc)
    ]
    return np.vstack(chunks)


def noise_std_for_cos(rho, dim):
    """Per-dimension Gaussian std s so that E[cos(target, target+noise)] ~ rho.
    E[cos] ~ 1/sqrt(1 + dim*s^2)  =>  s = sqrt(1/rho^2 - 1)/sqrt(dim)."""
    return np.sqrt(max(1.0 / rho ** 2 - 1.0, 0.0)) / np.sqrt(dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "all-mpnet-base-v2"))
    ap.add_argument("--n-dict", type=int, default=5000)
    ap.add_argument("--n-query", type=int, default=1000)
    ap.add_argument("--cos-levels", type=float, nargs="+", default=[0.95, 0.8, 0.6],
                    help="target cosine between 'decoded' and target embedding")
    ap.add_argument("--examples", type=int, default=5)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    print("loading model:", args.model)
    model = SentenceTransformer(args.model, device="cpu")
    print("  -> embedding dim:", model.get_sentence_embedding_dimension())

    images = load_captions()
    ids = sorted(images.keys())
    dict_ids, query_ids = ids[:args.n_dict], ids[-args.n_query:]
    assert not (set(dict_ids) & set(query_ids)), "dict/query overlap!"

    cache = OUT / f"embed_cache_{len(dict_ids)}_{len(query_ids)}.npz"
    if cache.exists():
        print("loading cached embeddings:", cache.name)
        z = np.load(cache)
        D, T, query_ids = z["D"], z["T"], list(z["query_ids"])
        dict_owner = z["dict_owner"]
        dict_caps = list(z["dict_caps"])
        n_distract = int(z["n_distract"])
    else:
        # ---- dictionary: distractor captions (dict images) + the query ------
        # images' own captions. A "hit" = retrieving a caption belonging to the
        # query image among thousands of same-domain distractors.
        dict_caps, dict_owner = [], []
        for i in dict_ids:
            for c in images[i]:
                dict_caps.append(c)
                dict_owner.append(i)
        n_distract = len(dict_caps)
        for i in query_ids:                      # target captions go in as well
            for c in images[i][:5]:
                dict_caps.append(c)
                dict_owner.append(i)
        D = embed_captions(model, dict_caps, desc="dict embed")
        dict_owner = np.asarray(dict_owner)
        # ---- queries: mean of the 5 captions, like the paper ----------------
        flat_caps, flat_owner = [], []
        for i in query_ids:
            for c in images[i][:5]:
                flat_caps.append(c)
                flat_owner.append(i)
        E = embed_captions(model, flat_caps, desc="query embed")
        T = np.stack([E[np.asarray(flat_owner) == i].mean(0) for i in query_ids])
        T /= np.linalg.norm(T, axis=1, keepdims=True)
        OUT.mkdir(exist_ok=True)
        np.savez(cache, D=D, T=T, query_ids=np.asarray(query_ids),
                 dict_owner=dict_owner, dict_caps=np.asarray(dict_caps, dtype=object),
                 n_distract=n_distract)
    print(f"dictionary: {n_distract} distractor captions "
          f"+ {len(dict_caps) - n_distract} target captions "
          f"= {len(dict_caps)} total")
    print(f"queries: {len(T)} target embeddings (mean of per-image captions)")

    rng = np.random.default_rng(0)
    rows = []
    for rho in [1.0] + list(args.cos_levels):
        if rho >= 1.0:
            P = T.copy()
        else:
            s = noise_std_for_cos(rho, D.shape[1])
            noise = rng.normal(0.0, s, size=T.shape).astype(np.float32)
            P = T + noise
            P /= np.linalg.norm(P, axis=1, keepdims=True)
        sim = P @ D.T                                  # cosine, queries x dict
        order = np.argsort(-sim, axis=1)
        best = np.full(len(P), np.inf)
        for q in range(len(P)):
            owned = np.where(dict_owner == query_ids[q])[0]
            best[q] = int(np.where(np.isin(order[q], owned))[0][0])
        cos_t = float(np.mean(np.sum(P * T, axis=1)))
        rows.append((rho, cos_t, float(np.mean(best == 0)),
                     float(np.mean(best < 5)), float(np.mean(best))))
        print(f"target_cos={rho:<5} empirical_cos={cos_t:.3f}  "
              f"hit@1={rows[-1][2]:.3f}  hit@5={rows[-1][3]:.3f}  "
              f"mean_best_rank={rows[-1][4]:.1f}")

    # ---- qualitative examples: distractor-only lookup at the lowest -----
    # cosine level, simulating what a real decoder would output
    distract_sim = P[:, :n_distract] @ D[:n_distract].T
    print(f"\n--- qualitative examples (target_cos = {args.cos_levels[-1]}, "
          f"distractor-only dictionary) ---")
    for q in range(min(args.examples, len(P))):
        print(f"\n[query image {query_ids[q]}] true captions:")
        for c in images[query_ids[q]][:3]:
            print("   *", c)
        print("  retrieved top-3 from dictionary:")
        for j in np.argsort(-distract_sim[q])[:3]:
            print("   -", dict_caps[j])

    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / "step1_results.npz",
                        cos_target=[r[0] for r in rows],
                        cos_empirical=[r[1] for r in rows],
                        hit1=[r[2] for r in rows],
                        hit5=[r[3] for r in rows],
                        mean_best_rank=[r[4] for r in rows])
    print("\nsaved -> outputs/step1_results.npz")


if __name__ == "__main__":
    main()
