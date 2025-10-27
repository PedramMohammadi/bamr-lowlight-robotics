# bamr/utils.py
from __future__ import annotations
import random, time
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.nn as nn

def seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# AMP helper compatible with torch>=2.0 and older torch.cuda.amp
try:
    from torch.amp import autocast as _autocast_new
    def amp_autocast(device: str = "cuda"):
        return _autocast_new(device, dtype=torch.float16) if device == "cuda" else _nullcontext()
except Exception:
    from torch.cuda.amp import autocast as _autocast_old
    def amp_autocast(device: str = "cuda"):
        return _autocast_old(dtype=torch.float16, enabled=(device == "cuda")) if device == "cuda" else _nullcontext()

class _nullcontext:
    def __enter__(self): return None
    def __exit__(self, *args): return False

def save_checkpoint(path: str, model: nn.Module, extra: Optional[Dict[str,Any]] = None):
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sd = model.state_dict()
    pkg = {"model": sd}
    if extra:
        pkg.update(extra)
    torch.save(pkg, path)
    print(f"[*] Saved checkpoint -> {path}")

def load_checkpoint(path: str, map_location="cpu") -> Dict[str,Any]:
    return torch.load(path, map_location=map_location)

class AvgMeter:
    def __init__(self):
        self.n = 0; self.sum = 0.0
    def update(self, v: float, k: int = 1):
        self.sum += float(v) * k; self.n += k
    @property
    def avg(self) -> float:
        return self.sum / max(1, self.n)

def psnr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """a,b in [0,1], returns PSNR in dB."""
    mse = torch.mean((a - b) ** 2)
    return -10.0 * torch.log10(mse + eps)

def tensor_to_u8(img: torch.Tensor) -> np.ndarray:
    # [3,H,W] or [B,3,H,W]
    if img.ndim == 4:
        img = img[0]
    x = (img.clamp(0,1) * 255.0).byte().cpu().permute(1,2,0).numpy()
    return x

def time_sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return time.time()
