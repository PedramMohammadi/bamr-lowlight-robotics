#!/usr/bin/env python3
"""
LoFTR “before vs after” matching on your MID mini subset.

- Loads TinyBAMR from Stage‑A/Stage‑B checkpoints (robust strict=False).
- Enhances both views with Stage‑A and Stage‑B, runs LoFTR (outdoor weights).
- Computes matches, mean confidence, and optional Sampson inliers with E_estimated if available.
- Downscales long side to MAX_SIDE to avoid OOM on L4/T4.

Outputs:
  reports/matching_eval/mid_loftr_before_after.csv
  and prints overall/per-domain summaries.
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np, pandas as pd, cv2, torch, torch.nn as nn
from kornia.feature import LoFTR
from tqdm import tqdm

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---- TinyBAMR micro-arch (same as used in your notebook patches) ----
class DSConvBlock(nn.Module):
    def __init__(self,in_ch,out_ch,k=3,s=1,act=True):
        super().__init__(); p=k//2
        self.dw=nn.Conv2d(in_ch,in_ch,k,s,p,groups=in_ch,bias=False)
        self.pw=nn.Conv2d(in_ch,out_ch,1,1,0,bias=False)
        self.bn=nn.BatchNorm2d(out_ch)
        self.act=nn.SiLU(inplace=True) if act else nn.Identity()
    def forward(self,x): x=self.dw(x); x=self.pw(x); x=self.bn(x); return self.act(x)

class ResDS(nn.Module):
    def __init__(self,ch):
        super().__init__(); self.b1=DSConvBlock(ch,ch); self.b2=DSConvBlock(ch,ch,act=False); self.act=nn.SiLU(inplace=True)
    def forward(self,x): return self.act(self.b2(self.b1(x))+x)

class LRFBlock(nn.Module):
    def __init__(self,ch,dilations=(1,2,4)):
        super().__init__()
        self.branches=nn.ModuleList([nn.Conv2d(ch,ch,3,1,d,groups=ch,bias=False,dilation=d) for d in dilations])
        self.proj=nn.Conv2d(ch*len(dilations),ch,1,1,0,bias=False)
        self.bn=nn.BatchNorm2d(ch); self.act=nn.SiLU(inplace=True)
    def forward(self,x):
        x=torch.cat([b(x) for b in self.branches],1); x=self.proj(x); x=self.bn(x); return self.act(x)

class FreqAttention(nn.Module):
    def __init__(self,ch,kernel=3):
        super().__init__(); pad=kernel//2
        self.blur=nn.Conv2d(ch,ch,kernel,1,pad,groups=ch,bias=False)
        with torch.no_grad():
            w=torch.zeros_like(self.blur.weight); w[:]=1.0/(kernel*kernel); self.blur.weight.copy_(w)
        hid=max(8,ch//8)
        self.se=nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch,hid,1), nn.SiLU(inplace=True), nn.Conv2d(hid,ch,1), nn.Sigmoid())
    def forward(self,x):
        low=self.blur(x); high=x-low; w=self.se(high); return x+high*w

class Up(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__(); self.up=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False); self.conv=DSConvBlock(in_ch,out_ch)
    def forward(self,x,skip): 
        x=self.up(x)
        # Safe-crop for odd shapes
        dh = min(x.shape[-2], skip.shape[-2]); dw = min(x.shape[-1], skip.shape[-1])
        x = x[..., :dh, :dw]; s = skip[..., :dh, :dw]
        x=torch.cat([x,s],1)
        return self.conv(x)

class TinyBAMR(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        ch1,ch2,ch3,ch4=base,base*2,base*4,base*8
        self.enc1=nn.Sequential(DSConvBlock(3,ch1), ResDS(ch1))
        self.enc2=nn.Sequential(DSConvBlock(ch1,ch2,s=2), ResDS(ch2))
        self.enc3=nn.Sequential(DSConvBlock(ch2,ch3,s=2), ResDS(ch3))
        self.enc4=nn.Sequential(DSConvBlock(ch3,ch4,s=2), ResDS(ch4))
        self.bot =nn.Sequential(ResDS(ch4), LRFBlock(ch4), FreqAttention(ch4))
        self.dec3=Up(ch4+ch3,ch3); self.dec2=Up(ch3+ch2,ch2); self.dec1=Up(ch2+ch1,ch1)
        self.out =nn.Sequential(DSConvBlock(ch1,ch1), nn.Conv2d(ch1,3,1), nn.Sigmoid())
    def forward(self,x):
        e1=self.enc1(x); e2=self.enc2(e1); e3=self.enc3(e2); e4=self.enc4(e3)
        b=self.bot(e4); d3=self.dec3(b,e3); d2=self.dec2(d3,e2); d1=self.dec1(d2,e1)
        return self.out(d1)

def load_bamr_tiny(ckpt_path: Path, device: str="cuda", half: bool=True) -> nn.Module:
    m = TinyBAMR(base=32)
    state = torch.load(str(ckpt_path), map_location="cpu")
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[INFO] load_bamr: missing={len(missing)} unexpected={len(unexpected)}")
    m.eval().to(device)
    if half and device=="cuda":
        m.half()
    return m

# ---- helpers ----
def resize_longest(rgb, max_side=1280):
    h, w = rgb.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return rgb
    scale = max_side / s
    nh, nw = int(round(h*scale)), int(round(w*scale))
    return cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)

def enhance(bamr: nn.Module, img_rgb_u8, device: str="cuda"):
    x = torch.from_numpy(img_rgb_u8).to(device)
    x = x.permute(2,0,1).unsqueeze(0)
    x = (x.half() if next(bamr.parameters()).dtype==torch.float16 else x.float())/255.0
    with torch.no_grad():
        y = bamr(x)
    y = (y.clamp(0,1)*255.0).squeeze(0).permute(1,2,0).cpu().numpy().astype(np.uint8)
    return y

def loftr_match(loftr: LoFTR, gray0, gray1, device="cuda"):
    # Inputs: uint8 gray
    t0 = torch.from_numpy(gray0)[None,None].float().to(device)/255.0
    t1 = torch.from_numpy(gray1)[None,None].float().to(device)/255.0
    with torch.no_grad():
        out = loftr({"image0": t0, "image1": t1})
    if ("keypoints0" not in out) or (out["keypoints0"].numel()==0):
        return np.empty((0,2)), np.empty((0,2)), np.empty((0,))
    mk0 = out["keypoints0"].detach().float().cpu().numpy()
    mk1 = out["keypoints1"].detach().float().cpu().numpy()
    conf = out["confidence"].detach().float().cpu().numpy()
    return mk0, mk1, conf

def sampson_inliers(mk0, mk1, E, thresh=1.0) -> int:
    if mk0.shape[0]==0 or E is None:
        return 0
    pts0 = cv2.convertPointsToHomogeneous(mk0).reshape(-1,3)
    pts1 = cv2.convertPointsToHomogeneous(mk1).reshape(-1,3)
    Ex1  = (E @ pts0.T).T
    ETx2 = (E.T @ pts1.T).T
    x2tEx1 = np.sum(pts1 * Ex1, axis=1)
    denom = Ex1[:,0]**2 + Ex1[:,1]**2 + ETx2[:,0]**2 + ETx2[:,1]**2 + 1e-6
    d = (x2tEx1**2) / denom
    return int((d < thresh).sum())

def pick_frames(viewA: Path, viewB: Path, K=3):
    A = sorted(list(viewA.glob("*.jpg")) + list(viewA.glob("*.JPG")))
    B = sorted(list(viewB.glob("*.jpg")) + list(viewB.glob("*.JPG")))
    if not A or not B: return []
    A_map = {p.stem: p for p in A}
    B_map = {p.stem: p for p in B}
    common = sorted(set(A_map).intersection(B_map))
    frames = [(A_map[s], B_map[s]) for s in common] if common else list(zip(A,B))
    if len(frames) <= K: return frames
    idxs = [0, len(frames)//2, len(frames)-1]
    return [frames[i] for i in idxs]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default="/content/drive/MyDrive/bamr_project")
    ap.add_argument("--stageA_ckpt", type=str, default="checkpoints/bamr_stageA/bamr_stageA_best.pt")
    ap.add_argument("--stageB_ckpt", type=str, default="checkpoints/bamr_stageB/bamr_taskaware.pt")
    ap.add_argument("--max_side", type=int, default=1280)
    ap.add_argument("--pairs_k", type=int, default=3)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    PR = Path(args.project_root)
    MID_IMG = PR / "data" / "prepared" / "mid_mini" / "images"
    MID_RAW = PR / "data" / "raw" / "mid" / "mini"
    OUT_DIR = PR / "reports" / "matching_eval"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV = OUT_DIR / "mid_loftr_before_after.csv"

    device = args.device if (args.device=="cpu" or torch.cuda.is_available()) else "cpu"

    bamrA = load_bamr_tiny(PR/args.stageA_ckpt, device=device, half=(device=="cuda"))
    bamrB = load_bamr_tiny(PR/args.stageB_ckpt, device=device, half=(device=="cuda"))
    print("Loaded:", type(bamrA).__name__, type(bamrB).__name__)

    # LoFTR: try GPU half; if OOM, fall back to CPU float
    def get_loftr(dev):
        m = LoFTR(pretrained="outdoor").to(dev).eval()
        if dev == "cuda":
            m = m.half()
        return m
    loftr = None
    try:
        loftr = get_loftr(device)
    except Exception as e:
        print("[WARN] LoFTR GPU init failed, using CPU:", e)
        device = "cpu"
        loftr = get_loftr(device)

    rows = []
    total_pairs = 0
    for domain in ("Indoor","Outdoor"):
        for pair_dir in sorted((MID_IMG/domain).glob("pair*")):
            vA = pair_dir/"viewA"; vB = pair_dir/"viewB"
            if not vA.exists() or not vB.exists(): 
                continue
            total_pairs += 1
            E_path = MID_RAW/domain/pair_dir.name/"GT_Correspondence"/"E_estimated.npy"
            E = np.load(str(E_path)) if E_path.exists() else None
            for (pa,pb) in pick_frames(vA,vB,K=args.pairs_k):
                A0 = cv2.cvtColor(cv2.imread(str(pa)), cv2.COLOR_BGR2RGB)
                B0 = cv2.cvtColor(cv2.imread(str(pb)), cv2.COLOR_BGR2RGB)
                for tag, net in (("orig",None),("stageA",bamrA),("stageB",bamrB)):
                    A = A0 if net is None else enhance(net, A0, device=device)
                    B = B0 if net is None else enhance(net, B0, device=device)
                    A = resize_longest(A, args.max_side); B = resize_longest(B, args.max_side)
                    gA = cv2.cvtColor(A, cv2.COLOR_RGB2GRAY)
                    gB = cv2.cvtColor(B, cv2.COLOR_RGB2GRAY)
                    try:
                        mk0, mk1, conf = loftr_match(loftr, gA, gB, device=device)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower() and device=="cuda":
                            print("[INFO] LoFTR CUDA OOM → CPU fallback for remainder.")
                            device = "cpu"
                            loftr = get_loftr(device)
                            mk0, mk1, conf = loftr_match(loftr, gA, gB, device=device)
                        else:
                            raise
                    n = int(conf.size); mc = float(conf.mean()) if n>0 else 0.0
                    inl = sampson_inliers(mk0, mk1, E, thresh=1.0) if (E is not None and n>0) else np.nan
                    rows.append({
                        "domain": domain, "pair": pair_dir.name,
                        "viewA": pa.name, "viewB": pb.name,
                        "cond": tag, "matches": n, "mean_conf": mc, "inliers": inl
                    })

    df = pd.DataFrame(rows)
    df.to_csv(CSV, index=False)
    print("Wrote:", CSV)

    def summarize(_df):
        return _df.groupby("cond").agg(
            n_rows=("mean_conf","count"),
            mean_conf=("mean_conf","mean"),
            mean_matches=("matches","mean"),
            mean_inliers=("inliers","mean")
        ).reset_index().sort_values("cond")

    print("\n=== Overall summary ===")
    print(summarize(df))

    for dom in ("Indoor","Outdoor"):
        sdf = df[df["domain"]==dom]
        if len(sdf):
            print(f"\n[{dom}]")
            print(summarize(sdf))

    pairs_seen = df[["domain","pair"]].drop_duplicates().shape[0]
    print(f"\nProcessed ~{pairs_seen} pairs (requested: {total_pairs}) and {len(df)//3} frame-pairs per condition.")

if __name__ == "__main__":
    main()
