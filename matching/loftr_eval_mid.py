# -*- coding: utf-8 -*-
"""
matching/loftr_eval_mid.py
LoFTR “before vs after” on MID mini (orig vs Stage‑A vs Stage‑B), memory‑safe:
 - GPU fp16 with automatic fallback to CPU on OOM
 - Resizes long side to --max_side (default 1280)
 - Pads inputs to / unpads from multiples of 8 for U‑Net skip alignment
Outputs:
 - reports/matching_eval/mid_loftr_before_after.csv (per-frame rows)
 - Console summary (overall and per domain)
"""

import os
import sys
import argparse
import contextlib
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2

# --- Default config (can override via CLI)
DEFAULT_PROJECT_ROOT = "/content/drive/MyDrive/bamr_project"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ------------------------- AMP helper (torch version agnostic)
def _make_amp(device: str):
    try:
        from torch.amp import autocast as _autocast_new
        return (lambda: _autocast_new('cuda', dtype=torch.float16)) if device == "cuda" else (lambda: contextlib.nullcontext())
    except Exception:
        from torch.cuda.amp import autocast as _autocast_old
        return (lambda: _autocast_old(dtype=torch.float16, enabled=(device=="cuda"))) if device == "cuda" else (lambda: contextlib.nullcontext())

# ------------------------- Geometry helpers
def resize_longest(rgb_u8: np.ndarray, max_side: int) -> np.ndarray:
    h, w = rgb_u8.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return rgb_u8
    scale = max_side / s
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(rgb_u8, (nw, nh), interpolation=cv2.INTER_AREA)

def pad_to_multiple(rgb_u8: np.ndarray, multiple: int = 8) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    """Pad bottom/right so H,W are divisible by `multiple`. Returns padded image and (top, bottom, left, right) pad."""
    h, w = rgb_u8.shape[:2]
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return rgb_u8, (0,0,0,0)
    padded = cv2.copyMakeBorder(rgb_u8, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REFLECT_101)
    return padded, (0, pad_h, 0, pad_w)

def unpad(rgb_u8: np.ndarray, pads: Tuple[int,int,int,int]) -> np.ndarray:
    t,b,l,r = pads
    if (t,b,l,r) == (0,0,0,0):
        return rgb_u8
    h, w = rgb_u8.shape[:2]
    return rgb_u8[t:h-b if b>0 else h, l:w-r if r>0 else w]

# ------------------------- BAMR loader
def load_bamr_tiny(project_root: Path, ckpt_path: Path, device: str, half: bool = True) -> nn.Module:
    """Load TinyBAMR from bamr.models with flexible state_dict format."""
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from bamr.models import TinyBAMR
    except Exception as e:
        print(f"[WARN] Could not import bamr.models.TinyBAMR ({e}). Using identity.")
        return nn.Identity().to(device)

    m = TinyBAMR(base=32)
    try:
        state = torch.load(str(ckpt_path), map_location="cpu")
        sd = state["model"] if isinstance(state, dict) and "model" in state else state
        missing, unexp = m.load_state_dict(sd, strict=False)
        if missing or unexp:
            print(f"[INFO] BAMR load: missing={len(missing)} unexpected={len(unexp)} (buffers/BN are OK).")
    except Exception as e:
        print(f"[WARN] Could not load {ckpt_path} ({e}). Using identity.")
        return nn.Identity().to(device)

    m.eval().to(device)
    if half and device == "cuda":
        m.half()
    return m

def enhance_with_bamr(img_rgb_u8: np.ndarray, model: nn.Module, device: str, amp_ctx, pad_mult: int = 8) -> np.ndarray:
    """Run BAMR safely: pad to stride, run (fp16 on cuda), unpad back."""
    padded, pads = pad_to_multiple(img_rgb_u8, pad_mult)
    t = torch.from_numpy(padded).to(device)
    # match dtype to model
    dtype = next(model.parameters()).dtype if any(p.requires_grad or p.is_floating_point() for p in model.parameters()) else torch.float32
    t = t.permute(2,0,1).unsqueeze(0)  # 1,3,H,W
    t = (t.to(dtype) / 255.0) if dtype in (torch.float16, torch.float32, torch.bfloat16) else t.float().div(255.0)
    with torch.no_grad(), amp_ctx():
        y = model(t) if not isinstance(model, nn.Identity) else t
    y = (y.clamp(0,1)*255.0).squeeze(0).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
    return unpad(y, pads)

# ------------------------- LoFTR (GPU with CPU fallback)
class LoFTRRunner:
    def __init__(self, device: str):
        self.device = device
        self.USE_CPU_FALLBACK = False
        self._gpu = None
        self._cpu = None

    def _get_gpu(self):
        if self._gpu is None:
            from kornia.feature import LoFTR
            self._gpu = LoFTR(pretrained="outdoor").to("cuda").eval().half()
        return self._gpu

    def _get_cpu(self):
        if self._cpu is None:
            from kornia.feature import LoFTR
            self._cpu = LoFTR(pretrained="outdoor").to("cpu").eval()
        return self._cpu

    def match(self, gray0_u8: np.ndarray, gray1_u8: np.ndarray, amp_ctx) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        if self.USE_CPU_FALLBACK or self.device != "cuda":
            t0 = torch.from_numpy(gray0_u8)[None,None].float().cpu().div(255.0)
            t1 = torch.from_numpy(gray1_u8)[None,None].float().cpu().div(255.0)
            with torch.no_grad():
                out = self._get_cpu()({"image0": t0, "image1": t1})
        else:
            t0 = torch.from_numpy(gray0_u8)[None,None].float().cuda().div(255.0).half()
            t1 = torch.from_numpy(gray1_u8)[None,None].float().cuda().div(255.0).half()
            try:
                with torch.no_grad(), amp_ctx():
                    out = self._get_gpu()({"image0": t0, "image1": t1})
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("[INFO] LoFTR OOM on CUDA → switching to CPU for remainder.")
                    self.USE_CPU_FALLBACK = True
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    t0 = t0.float().cpu(); t1 = t1.float().cpu()
                    with torch.no_grad():
                        out = self._get_cpu()({"image0": t0, "image1": t1})
                else:
                    raise
        if ("keypoints0" not in out) or (out["keypoints0"].numel() == 0):
            return np.empty((0,2)), np.empty((0,2)), np.empty((0,))
        mk0 = out["keypoints0"].detach().float().cpu().numpy()
        mk1 = out["keypoints1"].detach().float().cpu().numpy()
        conf = out["confidence"].detach().float().cpu().numpy()
        return mk0, mk1, conf

# ------------------------- Inlier check (Sampson distance on E)
def sampson_inliers(mk0: np.ndarray, mk1: np.ndarray, E: np.ndarray, thresh: float = 1.0) -> int:
    """Counts matches with Sampson distance < thresh. (Note: using pixel coords with provided E)."""
    if mk0.shape[0] == 0 or E is None:
        return 0
    pts0 = cv2.convertPointsToHomogeneous(mk0).reshape(-1,3)
    pts1 = cv2.convertPointsToHomogeneous(mk1).reshape(-1,3)
    Ex1  = (E @ pts0.T).T
    ETx2 = (E.T @ pts1.T).T
    x2tEx1 = np.sum(pts1 * Ex1, axis=1)
    denom = Ex1[:,0]**2 + Ex1[:,1]**2 + ETx2[:,0]**2 + ETx2[:,1]**2 + 1e-6
    d = (x2tEx1**2) / denom
    return int((d < thresh).sum())

# ------------------------- Frame picking
def pick_frames(viewA_dir: Path, viewB_dir: Path, k: int = 3) -> List[Tuple[Path,Path]]:
    A = sorted([*viewA_dir.glob("*.jpg"), *viewA_dir.glob("*.JPG")])
    B = sorted([*viewB_dir.glob("*.jpg"), *viewB_dir.glob("*.JPG")])
    if not A or not B:
        return []
    A_stems = {p.stem: p for p in A}
    B_stems = {p.stem: p for p in B}
    common = sorted(set(A_stems).intersection(B_stems))
    frames = [(A_stems[s], B_stems[s]) for s in common] if common else list(zip(A, B))
    if len(frames) <= k:
        return frames
    idxs = [0, len(frames)//2, -1] if k == 3 else np.linspace(0, len(frames)-1, k).astype(int).tolist()
    return [frames[i] for i in idxs]

# ------------------------- Main
def run_eval(
    project_root: Path,
    ckpt_a: Path,
    ckpt_b: Path,
    max_side: int,
    frames_per_pair: int,
    out_csv: Path,
) -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = _make_amp(device)

    # Load enhancers
    bamrA = load_bamr_tiny(project_root, ckpt_a, device=device, half=True)
    bamrB = load_bamr_tiny(project_root, ckpt_b, device=device, half=True)

    # LoFTR runner
    loftr = LoFTRRunner(device)

    img_root = project_root / "data" / "prepared" / "mid_mini" / "images"
    raw_root = project_root / "data" / "raw" / "mid" / "mini"

    rows = []
    pairs_total = 0
    for domain in ["Indoor", "Outdoor"]:
        for pair_dir in sorted((img_root / domain).glob("pair*")):
            vA, vB = pair_dir/"viewA", pair_dir/"viewB"
            if not vA.exists() or not vB.exists():
                continue
            pairs_total += 1
            E_path = raw_root / domain / pair_dir.name / "GT_Correspondence" / "E_estimated.npy"
            E = np.load(str(E_path)) if E_path.exists() else None
            for pa, pb in pick_frames(vA, vB, frames_per_pair):
                A0 = cv2.cvtColor(cv2.imread(str(pa)), cv2.COLOR_BGR2RGB)
                B0 = cv2.cvtColor(cv2.imread(str(pb)), cv2.COLOR_BGR2RGB)
                for tag, model in (("orig", nn.Identity().to(device)), ("stageA", bamrA), ("stageB", bamrB)):
                    A1 = resize_longest(A0, max_side)
                    B1 = resize_longest(B0, max_side)
                    A = enhance_with_bamr(A1, model, device, amp)
                    B = enhance_with_bamr(B1, model, device, amp)
                    gA = cv2.cvtColor(A, cv2.COLOR_RGB2GRAY)
                    gB = cv2.cvtColor(B, cv2.COLOR_RGB2GRAY)
                    mk0, mk1, conf = loftr.match(gA, gB, amp)
                    n = int(conf.size)
                    mc = float(conf.mean()) if n > 0 else 0.0
                    inl = sampson_inliers(mk0, mk1, E, 1.0) if (E is not None and n > 0) else np.nan
                    rows.append({
                        "domain": domain, "pair": pair_dir.name, "viewA": pa.name, "viewB": pb.name,
                        "cond": tag, "matches": n, "mean_conf": mc, "inliers": inl
                    })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("Wrote:", out_csv)

    def summarize(_df: pd.DataFrame) -> pd.DataFrame:
        return _df.groupby("cond").agg(
            n_rows=("mean_conf","count"),
            mean_conf=("mean_conf","mean"),
            mean_matches=("matches","mean"),
            mean_inliers=("inliers","mean")
        ).reset_index().sort_values("cond")

    print("\n=== Overall summary ===")
    overall = summarize(df)
    print(overall)

    for dom in ["Indoor","Outdoor"]:
        sdf = df[df["domain"] == dom]
        if len(sdf):
            print(f"\n[{dom}]")
            print(summarize(sdf))

    pairs_seen = df[["domain","pair"]].drop_duplicates().shape[0]
    print(f"\nProcessed ~{pairs_seen} pairs (requested: {pairs_total}) and {len(df)//3} frame‑pairs per condition.")

    return {"overall": overall.to_dict(orient="list"), "rows": len(df)}

def parse_args():
    ap = argparse.ArgumentParser(description="LoFTR MID mini evaluation (orig vs Stage‑A vs Stage‑B).")
    ap.add_argument("--project_root", type=str, default=DEFAULT_PROJECT_ROOT)
    ap.add_argument("--ckpt_a", type=str, default="checkpoints/bamr_stageA/bamr_stageA_best.pt")
    ap.add_argument("--ckpt_b", type=str, default="checkpoints/bamr_stageB/bamr_taskaware.pt")
    ap.add_argument("--max_side", type=int, default=1280)
    ap.add_argument("--frames_per_pair", type=int, default=3)
    ap.add_argument("--out_csv", type=str, default="reports/matching_eval/mid_loftr_before_after.csv")
    return ap.parse_args()

def main():
    args = parse_args()
    project_root = Path(args.project_root)
    ckpt_a = project_root / args.ckpt_a if not args.ckpt_a.startswith(("/", ".")) else Path(args.ckpt_a)
    ckpt_b = project_root / args.ckpt_b if not args.ckpt_b.startswith(("/", ".")) else Path(args.ckpt_b)
    out_csv = project_root / args.out_csv if not args.out_csv.startswith(("/", ".")) else Path(args.out_csv)

    run_eval(project_root, ckpt_a, ckpt_b, args.max_side, args.frames_per_pair, out_csv)

if __name__ == "__main__":
    main()
