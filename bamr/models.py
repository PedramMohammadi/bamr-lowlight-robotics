# bamr/models.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# ---- Blocks ----
class DSConvBlock(nn.Module):
    """Depthwise separable conv block with SiLU + BN."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, act: bool = True):
        super().__init__()
        p = k // 2
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)

class ResDS(nn.Module):
    """Depthwise residual block."""
    def __init__(self, ch: int):
        super().__init__()
        self.b1 = DSConvBlock(ch, ch)
        self.b2 = DSConvBlock(ch, ch, act=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.b2(self.b1(x)) + x)

class LRFBlock(nn.Module):
    """Long-receptive-field via dilated depthwise branches + 1x1 fuse."""
    def __init__(self, ch: int, dilations=(1,2,4)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(ch, ch, 3, 1, d, groups=ch, dilation=d, bias=False) for d in dilations
        ])
        self.proj = nn.Conv2d(ch * len(dilations), ch, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([b(x) for b in self.branches], dim=1)
        x = self.proj(x)
        x = self.bn(x)
        return self.act(x)

class FreqAttention(nn.Module):
    """Simple frequency attention: blur -> high-freq -> SE gate."""
    def __init__(self, ch: int, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.blur = nn.Conv2d(ch, ch, kernel, 1, pad, groups=ch, bias=False)
        with torch.no_grad():
            w = torch.zeros_like(self.blur.weight)
            w[:] = 1.0 / (kernel * kernel)
            self.blur.weight.copy_(w)
        hid = max(8, ch // 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, hid, 1), nn.SiLU(inplace=True),
            nn.Conv2d(hid, ch, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = self.blur(x)
        high = x - low
        w = self.se(high)
        return x + high * w

class Up(nn.Module):
    """Upsample to skip size and fuse."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = DSConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

def _pad_to_multiple(x: torch.Tensor, m: int = 8, mode: str = "reflect") -> Tuple[torch.Tensor, Tuple[int,int]]:
    """Pad (right/bottom) to next multiple of m."""
    B, C, H, W = x.shape
    ph = (m - H % m) % m
    pw = (m - W % m) % m
    if ph == 0 and pw == 0:
        return x, (H, W)
    x = F.pad(x, (0, pw, 0, ph), mode=mode)
    return x, (H, W)

# ---- TinyBAMR ----
class TinyBAMR(nn.Module):
    """Lightweight enhancer with shape guards (pad to /8, crop back)."""
    def __init__(self, base: int = 32):
        super().__init__()
        c1, c2, c3, c4 = base, base*2, base*4, base*8
        self.enc1 = nn.Sequential(DSConvBlock(3, c1), ResDS(c1))
        self.enc2 = nn.Sequential(DSConvBlock(c1, c2, s=2), ResDS(c2))
        self.enc3 = nn.Sequential(DSConvBlock(c2, c3, s=2), ResDS(c3))
        self.enc4 = nn.Sequential(DSConvBlock(c3, c4, s=2), ResDS(c4))
        self.bot  = nn.Sequential(ResDS(c4), LRFBlock(c4), FreqAttention(c4))
        self.dec3 = Up(c4 + c3, c3)
        self.dec2 = Up(c3 + c2, c2)
        self.dec1 = Up(c2 + c1, c1)
        self.out  = nn.Sequential(DSConvBlock(c1, c1), nn.Conv2d(c1, 3, 1), nn.Sigmoid())

    @torch.no_grad()
    def _sanity(self):
        x = torch.randn(1,3,255,257)
        y = self.forward(x)
        assert y.shape[-2:] == (255,257)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, (H, W) = _pad_to_multiple(x, m=8, mode="reflect")
        e1 = self.enc1(x); e2 = self.enc2(e1); e3 = self.enc3(e2); e4 = self.enc4(e3)
        b  = self.bot(e4)
        d3 = self.dec3(b, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        y  = self.out(d1)
        return y[..., :H, :W]

def load_bamr_tiny(ckpt_path: str, device: str = "cuda", half: bool = True) -> TinyBAMR:
    """Load TinyBAMR from a Stage‑A/Stage‑B checkpoint (either raw sd or {'model': sd})."""
    m = TinyBAMR(base=32)
    state = torch.load(ckpt_path, map_location="cpu")
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[INFO] load_bamr_tiny: missing={len(missing)} unexpected={len(unexpected)}")
    m.eval().to(device)
    if half and device == "cuda":
        m.half()
    return m
