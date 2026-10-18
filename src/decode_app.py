# -*- coding: utf-8 -*-
"""
Interactive fMRI -> text decoding app (Tkinter, no extra dependencies).

Usage:
    python decode_app.py            # launch GUI
    python decode_app.py --selftest # headless test of the decode logic

Workflow: pick a trained ridge model, pick an image the subject saw
(its captions are shown as ground truth), press 开始解码 — the app loads
that image's pre-extracted fMRI beta vectors (with session/timepoint
provenance), predicts the 768-d MPNet embedding via the ridge model, then
looks up the nearest captions in the dictionary and shows the top-5 matches
with cosine scores.
"""
import argparse
import os
import sys
import tkinter as tk
from tkinter import ttk
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FALLBACK_ROOTS = [Path(r"D:/codex project/test/visuo_llm_repro")]


def find_root():
    """Locate the working dir that contains data/, models/, outputs/."""
    cand = [Path(os.environ["VISUO_ROOT"])] if os.environ.get("VISUO_ROOT") else []
    p = Path(__file__).resolve().parent
    for _ in range(4):
        cand.append(p); p = p.parent
    cand += FALLBACK_ROOTS
    for c in cand:
        if (c / "outputs" / "ridge_models").exists():
            return c
    raise SystemExit("找不到数据根目录：请设置环境变量 VISUO_ROOT 指向 visuo_llm_repro")


class DecodingApp:
    def __init__(self, root):
        self.root = root
        self.bundle = None
        self.dict_D = self.dict_caps = self.dict_owner = None
        self.Y_all = self.nsd_ids = None

        root.title("fMRI → 视觉语义解码器 (NSD subj01 · MPNet · 岭回归)")
        root.geometry("1180x800")

        top = ttk.Frame(root, padding=8); top.pack(fill="x")
        ttk.Label(top, text="岭回归模型:").pack(side="left")
        self.model_var = tk.StringVar()
        self.model_box = ttk.Combobox(top, textvariable=self.model_var,
                                      width=52, state="readonly")
        self.model_box.pack(side="left", padx=4)
        self.model_box.bind("<<ComboboxSelected>>", self.load_model)
        ttk.Button(top, text="刷新", command=self.list_models).pack(side="left")

        ttk.Label(top, text="  被试看过的图片:").pack(side="left")
        self.img_var = tk.StringVar()
        self.img_box = ttk.Combobox(top, textvariable=self.img_var,
                                    width=64, state="readonly")
        self.img_box.pack(side="left", padx=4)
        self.img_box.bind("<<ComboboxSelected>>",
                          lambda e: self.show_image())

        self.go_btn = ttk.Button(top, text="开始解码 ▶", command=self.decode)
        self.go_btn.pack(side="left", padx=8)
        self.status = ttk.Label(top, text="请选择模型", foreground="#555")
        self.status.pack(side="left", padx=6)

        self._photo = None            # keep reference, avoid GC
        self.split_of = None

        pane = ttk.Frame(root, padding=8); pane.pack(fill="both", expand=True)
        pane.columnconfigure(1, weight=1); pane.rowconfigure(1, weight=1)

        # left column: the actual image + ground-truth captions
        left = ttk.Frame(pane)
        left.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 8))
        self.img_label = ttk.Label(left, anchor="center", width=48,
                                   text="（选择图片后在此显示）")
        self.img_label.pack(pady=(0, 4))
        self.gt_text = tk.Text(left, width=46, height=12, wrap="word",
                               font=("Microsoft YaHei UI", 10))
        self.gt_text.pack(fill="both", expand=True)

        # right column: decode results
        ttk.Label(pane, text="解码结果（字典最近邻 Top-5）").grid(
            row=0, column=1, sticky="w")
        self.res_text = tk.Text(pane, width=56, height=24, wrap="word",
                                font=("Microsoft YaHei UI", 10))
        self.res_text.grid(row=1, column=1, sticky="nsew")

        ttk.Label(root, text="解码诊断", padding=(10, 0)).pack(anchor="w")
        self.diag_text = tk.Text(root, height=7, wrap="word",
                                 font=("Consolas", 10))
        self.diag_text.pack(fill="x", padx=10, pady=(0, 10))

        self.list_models()

    # ---------- data loading -------------------------------------------
    def list_models(self):
        models = sorted((find_root() / "outputs" / "ridge_models").glob("*.npz"))
        if not models:
            self.status.config(text="outputs/ridge_models/ 下没有模型，先跑 train_model.py")
            return
        self.model_paths = models
        self.model_box["values"] = [m.name for m in models]
        self.model_box.current(0)
        self.load_model()

    def load_model(self, *evt):
        path = self.model_paths[self.model_box.current()]
        self.status.config(text=f"载入模型 {path.name} ...")
        self.root.update()
        z = np.load(path, allow_pickle=True)
        self.bundle = {k: z[k] for k in z.files}
        n_img = len(self.bundle["img_nsdId"])
        labels = []
        for i in range(n_img):
            cap = next((c for c in self.bundle["img_caps"][i] if c), "(无描述)")
            labels.append(f"{i:04d} | coco {self.bundle['img_cocoId'][i]} | {cap[:52]}")
        self.img_box["values"] = labels
        self.img_box.current(0)
        info = (f"模型: {path.name} | {n_img} 张测试图 | "
                f"训练 CV cosine {float(self.bundle['cv_cosine']):.3f} | "
                f"测试 cosine {float(self.bundle['test_cos_mean']):.3f}")
        self.status.config(text=info)
        self.gt_text.delete("1.0", "end"); self.res_text.delete("1.0", "end")
        self.diag_text.delete("1.0", "end")

    def ensure_splits(self):
        """nsdId -> cocoSplit mapping (for locating image files)."""
        if self.split_of is not None:
            return
        z = np.load(find_root() / "outputs" / "nsd_subj01_caption_embeddings.npz",
                    allow_pickle=True)
        self.split_of = {int(n): str(s)
                         for n, s in zip(z["nsdId"], z["cocoSplit"])}

    def show_image(self, img_idx=None):
        """Display the actual COCO image for the selected test image."""
        if self.bundle is None:
            return
        if img_idx is None:
            img_idx = self.img_box.current()
        from PIL import Image, ImageTk
        self.ensure_splits()
        b = self.bundle
        coco = int(b["img_cocoId"][img_idx])
        split = self.split_of.get(int(b["img_nsdId"][img_idx]), "train2017")
        path = (find_root() / "data" / "coco" / "images" / split
                / f"{coco:012d}.jpg")
        if not path.exists():
            self.img_label.configure(
                image="", text="（该图片未下载\n运行 src/fetch_test_images.py）")
            self._photo = None
            return
        img = Image.open(path)
        img.thumbnail((350, 350))
        self._photo = ImageTk.PhotoImage(img)
        self.img_label.configure(image=self._photo, text="")

    def ensure_dict(self):
        if self.dict_D is not None:
            return
        self.status.config(text="载入字典（首次约 5 秒）...")
        self.root.update()
        root = find_root()
        z = np.load(root / "outputs" / "embed_cache_5000_1000.npz", allow_pickle=True)
        self.dict_D = z["D"]
        self.dict_caps = z["dict_caps"]
        self.dict_owner = z["dict_owner"]
        z2 = np.load(root / "outputs" / "nsd_subj01_caption_embeddings.npz",
                     allow_pickle=True)
        self.Y_all, self.nsd_ids = z2["Y"], z2["nsdId"]

    # ---------- decoding -------------------------------------------------
    def decode(self):
        if self.bundle is None:
            return
        img_idx = self.img_box.current()
        self.show_image(img_idx)
        self.ensure_dict()
        out = run_decode(self.bundle, img_idx, self.dict_D, self.dict_caps,
                         self.dict_owner, self.Y_all, self.nsd_ids)

        self.gt_text.delete("1.0", "end")
        self.gt_text.insert("end", f"图片 nsdId {out['nsdId']} "
                                   f"(COCO {out['cocoId']})\n")
        self.gt_text.insert("end", f"fMRI 信号来源: " +
                            ", ".join(f"session{s:02d}#t{t:03d}"
                                      for s, t in out["provenance"]) + "\n")
        self.gt_text.insert("end", "―" * 40 + "\n\n")
        for i, c in enumerate(out["captions"], 1):
            if c:
                self.gt_text.insert("end", f"  {i}. {c}\n")

        self.res_text.delete("1.0", "end")
        self.res_text.insert("end", "预测嵌入 → 字典最近邻:\n\n")
        for rank, (cap, sim, own) in enumerate(out["top5"], 1):
            tag = "  ★该图自身描述!" if own else ""
            self.res_text.insert("end", f"  {rank}. [{sim:+.3f}] {cap}{tag}\n\n")

        self.diag_text.delete("1.0", "end")
        self.diag_text.insert("end", out["diagnostics"])
        self.status.config(text="解码完成 ✓")

    # ---------- UI helpers ------------------------------------------
    def show(self, widget, text):  # (unused helper kept minimal)
        widget.delete("1.0", "end")
        widget.insert("end", text)


def run_decode(bundle, img_idx, D, dict_caps, dict_owner, Y_all, nsd_ids):
    """Core logic, separated from the GUI for headless testing."""
    b = bundle
    sel = np.where(b["te_img"] == img_idx)[0]
    X = b["Xte"][sel]                                  # (n_repeats, n_vox)
    Xz = (X - b["mu"]) / b["sd"]
    P = Xz @ b["W"]                                    # (n_repeats, 768)
    Pavg = P.mean(0)
    Pavg = Pavg / (np.linalg.norm(Pavg) + 1e-9)

    sims = Pavg @ D.T
    top = np.argsort(-sims)[:5]
    coco = int(b["img_cocoId"][img_idx])
    top5 = [(str(dict_caps[j]), float(sims[j]),
             int(dict_owner[j]) == coco) for j in top]

    tgt = b["te_Y"][sel].mean(0)
    tgt = tgt / np.linalg.norm(tgt)
    cos_tgt = float(Pavg @ tgt)
    per_rep = [float(p @ tgt / (np.linalg.norm(p) + 1e-9)) for p in P]

    yrow = int(np.where(nsd_ids == b["img_nsdId"][img_idx])[0][0])
    rank = int((Pavg @ Y_all.T > cos_tgt).sum())

    diag = (f"解码 vs 真值嵌入  余弦: {cos_tgt:+.3f}   "
            f"(逐次重复: {', '.join(f'{c:+.3f}' for c in per_rep)})\n"
            f"真值图在 10,000 张候选图中的排名: #{rank+1} (随机期望 ~#5000)\n"
            f"模型: {len(b['sessions'])} 个 session 训练, "
            f"lambda={float(b['best_lambda'])}, {int(b['n_vox'])} 体素, "
            f"训练于 {str(b['train_date'])}")
    return dict(nsdId=int(b["img_nsdId"][img_idx]), cocoId=coco,
                captions=[str(c) for c in b["img_caps"][img_idx]],
                provenance=[(int(b["te_session"][i]), int(b["te_tp"][i]))
                            for i in sel],
                top5=top5, cos_target=cos_tgt, per_repeat=per_rep,
                rank=rank, diagnostics=diag)


def selftest():
    root = find_root()
    models = sorted((root / "outputs" / "ridge_models").glob("*.npz"))
    assert models, "no trained model found"
    z = np.load(models[0], allow_pickle=True)
    bundle = {k: z[k] for k in z.files}
    zd = np.load(root / "outputs" / "embed_cache_5000_1000.npz", allow_pickle=True)
    zy = np.load(root / "outputs" / "nsd_subj01_caption_embeddings.npz",
                 allow_pickle=True)
    for i in range(3):
        out = run_decode(bundle, i, zd["D"], zd["dict_caps"], zd["dict_owner"],
                         zy["Y"], zy["nsdId"])
        print(f"[img {i}] nsdId={out['nsdId']} cos={out['cos_target']:+.3f} "
              f"rank #{out['rank']+1}")
        print("  GT  :", out["captions"][0][:70])
        print("  TOP1:", f"[{out['top5'][0][1]:+.3f}]", out["top5"][0][0][:70])
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        r = tk.Tk()
        DecodingApp(r)
        r.mainloop()
