# %% [markdown]
# # MID LoFTR “before vs after” (memory‑safe)
# Compares LoFTR matches & inliers on MID mini pairs with:
#   - original frames
#   - Stage‑A enhanced
#   - Stage‑B enhanced
# Uses the same TinyBAMR micro‑arch + half precision, and will fall back to CPU if LoFTR OOMs.

# %%
import os, sys, contextlib
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path
import numpy as np, pandas as pd, cv2, torch, torch.nn as nn

PROJECT_ROOT = Path("/content/drive/MyDrive/bamr_project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

CKPT_A = PROJECT_ROOT / "checkpoints/bamr_stageA/bamr_stageA_best.pt"
CKPT_B = PROJECT_ROOT / "checkpoints/bamr_stageB/bamr_taskaware.pt"

MID_IMG_ROOT = PROJECT_ROOT / "data/prepared/mid_mini/images"   # images/{Indoor|Outdoor}/pairXX/{viewA|viewB}/*.jpg
MID_RAW_ROOT = PROJECT_ROOT / "data/raw/mid/mini"               # .../pairXX/GT_Correspondence/E_estimated.npy
OUT_DIR = PROJECT_ROOT / "reports/matching_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SIDE = 1280

# %%
# AMP helper across torch versions
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    from torch.amp import autocast as _autocast_new
    def amp_cast(): return _autocast_new('cuda', dtype=torch.float16) if device=="cuda" else contextlib.nullcontext()
except Exception:
    from torch.cuda.amp import autocast as _autocast_old
    def amp_cast(): return _autocast_old(dtype=torch.float16, enabled=(device=="cuda")) if device=="cuda" else contextlib.nullcontext()

# %%
# TinyBAMR (same as Stage‑A/B)
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
        # pad to avoid odd/even mismatches before concat
        dh = skip.shape[-2] - x.shape[-2]
        dw = skip.shape[-1] - x.shape[-1]
        if dh or dw:
            x = nn.functional.pad(x, (0,max(0,dw), 0,max(0,dh)))
        x=torch.cat([x,skip],1)
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

def load_bamr_tiny(ckpt_path, device="cuda", half=True):
    m = TinyBAMR(base=32)
    state = torch.load(str(ckpt_path), map_location="cpu")
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[INFO] load_bamr_tiny: missing={len(missing)} unexpected={len(unexpected)}")
    m.eval().to(device)
    if half and device=="cuda":
        m.half()
    return m

bamrA = load_bamr_tiny(str(CKPT_A), device=device, half=True)
bamrB = load_bamr_tiny(str(CKPT_B), device=device, half=True)
print("Loaded:", bamrA.__class__.__name__, "|", bamrB.__class__.__name__, "| device:", device)

# %%
def resize_longest(rgb, max_side=MAX_SIDE):
    h, w = rgb.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return rgb
    scale = max_side / s
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)

def enhance(img_rgb_u8, which="orig"):
    x = torch.from_numpy(img_rgb_u8).to(device)
    x = x.permute(2,0,1).unsqueeze(0)  # [1,3,H,W]
    x = (x.half() if next(bamrA.parameters()).dtype == torch.float16 else x.float())/255.0
    with torch.no_grad(), amp_cast():
        if which == "stageA":
            y = bamrA(x)
        elif which == "stageB":
            y = bamrB(x)
        else:
            y = x
    y = (y.clamp(0,1)*255.0).squeeze(0).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
    return y

# %%
# LoFTR with GPU -> CPU fallback
!pip -q install kornia opencv-python-headless > /dev/null
from kornia.feature import LoFTR

USE_CPU_FALLBACK = False
loftr_gpu = None
loftr_cpu = None
if device == "cuda":
    loftr_gpu = LoFTR(pretrained="outdoor").to("cuda").eval()
    loftr_gpu = loftr_gpu.half()

def get_loftr_cpu():
    global loftr_cpu
    if loftr_cpu is None:
        loftr_cpu = LoFTR(pretrained="outdoor").to("cpu").eval()
    return loftr_cpu

def loftr_match(gray0, gray1):
    global USE_CPU_FALLBACK
    if USE_CPU_FALLBACK or device != "cuda":
        t0 = torch.from_numpy(gray0)[None,None].float().to("cpu")/255.0
        t1 = torch.from_numpy(gray1)[None,None].float().to("cpu")/255.0
        with torch.no_grad():
            out = get_loftr_cpu()({"image0": t0, "image1": t1})
    else:
        t0 = torch.from_numpy(gray0)[None,None].float().to("cuda")/255.0
        t1 = torch.from_numpy(gray1)[None,None].float().to("cuda")/255.0
        t0 = t0.half(); t1 = t1.half()
        try:
            with torch.no_grad(), amp_cast():
                out = loftr_gpu({"image0": t0, "image1": t1})
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("[INFO] LoFTR OOM on CUDA -> switching to CPU for the rest.")
                USE_CPU_FALLBACK = True
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                t0 = t0.float().cpu(); t1 = t1.float().cpu()
                with torch.no_grad():
                    out = get_loftr_cpu()({"image0": t0, "image1": t1})
            else:
                raise
    if ("keypoints0" not in out) or (out["keypoints0"].numel() == 0):
        return np.empty((0,2)), np.empty((0,2)), np.empty((0,))
    mkpts0 = out["keypoints0"].detach().float().cpu().numpy()
    mkpts1 = out["keypoints1"].detach().float().cpu().numpy()
    conf   = out["confidence"].detach().float().cpu().numpy()
    return mkpts0, mkpts1, conf

def sampson_inliers(mkpts0, mkpts1, E, thresh=1.0):
    if mkpts0.shape[0] == 0 or E is None:
        return 0
    pts0 = cv2.convertPointsToHomogeneous(mkpts0).reshape(-1,3)
    pts1 = cv2.convertPointsToHomogeneous(mkpts1).reshape(-1,3)
    Ex1  = (E @ pts0.T).T
    ETx2 = (E.T @ pts1.T).T
    x2tEx1 = np.sum(pts1 * Ex1, axis=1)
    denom = Ex1[:,0]**2 + Ex1[:,1]**2 + ETx2[:,0]**2 + ETx2[:,1]**2 + 1e-6
    d = (x2tEx1**2) / denom
    return int((d < thresh).sum())

def pick_frames(viewA_dir: Path, viewB_dir: Path, K=3):
    A = sorted([p for p in viewA_dir.glob("*.jpg")] + [p for p in viewA_dir.glob("*.JPG")])
    B = sorted([p for p in viewB_dir.glob("*.jpg")] + [p for p in viewB_dir.glob("*.JPG")])
    if not A or not B: return []
    A_stems = {p.stem:p for p in A}; B_stems = {p.stem:p for p in B}
    common = sorted(set(A_stems).intersection(B_stems))
    frames = [ (A_stems[s], B_stems[s]) for s in common ] if common else list(zip(A, B))
    if len(frames) <= K: return frames
    idxs = [0, len(frames)//2, -1]
    return [frames[i] for i in idxs]

# %%
rows = []
pairs_total = 0
for domain in ["Indoor","Outdoor"]:
    for pair_dir in sorted((MID_IMG_ROOT/domain).glob("pair*")):
        vA = pair_dir/"viewA"; vB = pair_dir/"viewB"
        if not vA.exists() or not vB.exists():
            continue
        pairs_total += 1
        E_path = (MID_RAW_ROOT/domain/pair_dir.name/"GT_Correspondence"/"E_estimated.npy")
        E = np.load(str(E_path)) if E_path.exists() else None
        frames = pick_frames(vA, vB, K=3)
        for (pa, pb) in frames:
            A0 = cv2.cvtColor(cv2.imread(str(pa)), cv2.COLOR_BGR2RGB)
            B0 = cv2.cvtColor(cv2.imread(str(pb)), cv2.COLOR_BGR2RGB)
            for tag in ["orig","stageA","stageB"]:
                A = resize_longest(enhance(A0, tag), MAX_SIDE)
                B = resize_longest(enhance(B0, tag), MAX_SIDE)
                gA = cv2.cvtColor(A, cv2.COLOR_RGB2GRAY)
                gB = cv2.cvtColor(B, cv2.COLOR_RGB2GRAY)
                mk0, mk1, conf = loftr_match(gA, gB)
                n = int(conf.size)
                mc = float(conf.mean()) if n>0 else 0.0
                inl = sampson_inliers(mk0, mk1, E, thresh=1.0) if (E is not None and n>0) else np.nan
                rows.append({
                    "domain": domain, "pair": pair_dir.name, "viewA": pa.name, "viewB": pb.name,
                    "cond": tag, "matches": n, "mean_conf": mc, "inliers": inl
                })
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

df = pd.DataFrame(rows)
csv_path = OUT_DIR / "mid_loftr_before_after.csv"
df.to_csv(csv_path, index=False)
print("Wrote:", csv_path)

def summarize(_df):
    return _df.groupby("cond").agg(
        n_rows=("mean_conf","count"),
        mean_conf=("mean_conf","mean"),
        mean_matches=("matches","mean"),
        mean_inliers=("inliers","mean")
    ).reset_index().sort_values("cond")

print("\n=== Overall summary ===")
overall = summarize(df)
print(overall)

print("\n=== Per-domain summary ===")
for dom in ["Indoor","Outdoor"]:
    sdf = df[df["domain"]==dom]
    if len(sdf):
        print(f"\n[{dom}]")
        print(summarize(sdf))

pairs_seen = df[["domain","pair"]].drop_duplicates().shape[0]
print(f"\nProcessed ~{pairs_seen} pairs (requested: {pairs_total}) and {len(df)//3} frame-pairs per condition.")
