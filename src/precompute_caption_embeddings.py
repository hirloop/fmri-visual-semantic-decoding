# -*- coding: utf-8 -*-
"""
Precompute the decoding target matrix Y for NSD subj01:
for each of the 10,000 images the subject saw, embed its ~5 COCO captions
with all-mpnet-base-v2 and average them (exactly the paper's recipe).

Output: outputs/nsd_subj01_caption_embeddings.npz with
    Y          (10000, 768) float32, L2-normalised
    nsdId      (10000,)     NSD stimulus id
    cocoId     (10000,)     COCO image id
    cocoSplit  (10000,)     'train2017' / 'val2017'
    n_caps     (10000,)     number of captions averaged
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"


def main():
    # 1. captions per cocoId (train + val)
    caps = {}
    for split in ["train", "val"]:
        d = json.loads((DATA / "coco" / f"captions_{split}2017.json").read_text(encoding="utf-8"))
        for a in d["annotations"]:
            caps.setdefault(a["image_id"], []).append(a["caption"].strip())
    print(f"captions loaded for {len(caps)} images")

    # 2. subj01's 10,000 images
    targets = []  # (nsdId, cocoId, split, [captions])
    with open(DATA / "nsd" / "nsddata" / "experiments" / "nsd" / "nsd_stim_info_merged.csv",
              encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["subject1"] == "1":
                targets.append((int(row["nsdId"]), int(row["cocoId"]),
                                row["cocoSplit"], caps.get(int(row["cocoId"]), [])))
    targets.sort()
    print(f"subj01 images: {len(targets)}, missing captions: "
          f"{sum(1 for _,_,_,c in targets if not c)}")

    # 3. flatten and embed everything in one pass
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    model = SentenceTransformer(str(ROOT / "models" / "all-mpnet-base-v2"), device="cpu")
    flat, owner = [], []
    for i, (_, _, _, cs) in enumerate(targets):
        for c in cs[:5]:
            flat.append(c)
            owner.append(i)
    print(f"embedding {len(flat)} captions on CPU (4 threads)...")
    E = model.encode(flat, batch_size=128, normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=True)
    E = np.asarray(E, dtype=np.float32)

    # 4. group-mean per image, normalise
    Y = np.zeros((len(targets), E.shape[1]), dtype=np.float32)
    owner = np.asarray(owner)
    for i in range(len(targets)):
        m = E[owner == i].mean(0)
        Y[i] = m / np.linalg.norm(m)
    n_caps = np.asarray([min(len(cs), 5) for _,_,_,cs in targets], dtype=np.int16)
    nsd_ids = np.asarray([t[0] for t in targets], dtype=np.int32)
    coco_ids = np.asarray([t[1] for t in targets], dtype=np.int32)
    splits = np.asarray([t[2] for t in targets])

    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / "nsd_subj01_caption_embeddings.npz",
                        Y=Y, nsdId=nsd_ids, cocoId=coco_ids,
                        cocoSplit=splits, n_caps=n_caps)
    print("saved -> outputs/nsd_subj01_caption_embeddings.npz")
    print(f"Y shape: {Y.shape}")


if __name__ == "__main__":
    main()
