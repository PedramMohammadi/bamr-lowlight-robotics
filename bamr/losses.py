# bamr/losses.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional perceptual dependencies
try:
    from torchvision import models
    _HAS_TV = True
except Exception:
    _HAS_TV = False

# Optional LoFTR dependency
try:
    from kornia.feature import LoFTR
    _HAS_LOFTR = True
except Exception:
    _HAS_LOFTR = False


# -------- Basic losses --------
class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.crit = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.crit(pred, target)


class EdgeLoss(nn.Module):
    """Edge preservation (Sobel) between two images."""
    def __init__(self, mode: str = "l1"):
        super().__init__()
        self.mode = mode

        sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3)
        sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer("kx", sobel_x)
        self.register_buffer("ky", sobel_y)

    def _edges(self, x: torch.Tensor) -> torch.Tensor:
        # grayscale
        if x.shape[1] == 3:
            xg = 0.2989*x[:,0:1] + 0.5870*x[:,1:2] + 0.1140*x[:,2:3]
        else:
            xg = x
        gx = F.conv2d(xg, self.kx, padding=1)
        gy = F.conv2d(xg, self.ky, padding=1)
        mag = torch.sqrt(gx*gx + gy*gy + 1e-8)
        return mag

    def forward(self, pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        e1 = self._edges(pred)
        e2 = self._edges(ref)
        if self.mode == "l1":
            return (e1 - e2).abs().mean()
        else:
            return F.mse_loss(e1, e2)


class TVLoss(nn.Module):
    """Total variation, encourages smoothness."""
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dy = (x[:,:,1:,:] - x[:,:,:-1,:]).abs()
        dx = (x[:,:,:,1:] - x[:,:,:,:-1]).abs()
        return self.weight * (dx.mean() + dy.mean())


# -------- Perceptual (VGG) --------
class _VGGFeatures(nn.Module):
    def __init__(self, layers=(2,7,12,21), requires_grad=False):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES if hasattr(models, "VGG16_Weights") else None).features
        # accumulate layers up to chosen indices
        blocks = []
        prev = 0
        for idx in layers:
            blocks.append(nn.Sequential(*[vgg[i] for i in range(prev, idx)]))
            prev = idx
        self.blocks = nn.ModuleList(blocks)
        if not requires_grad:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor):
        feats = []
        out = x
        for b in self.blocks:
            out = b(out)
            feats.append(out)
        return feats


class PerceptualLoss(nn.Module):
    """VGG-feature MSE between pred & target (expects [0,1])."""
    def __init__(self, weight: float = 1.0, layers=(2,7,12,21)):
        super().__init__()
        if not _HAS_TV:
            raise RuntimeError("torchvision not available for PerceptualLoss.")
        self.w = weight
        self.vgg = _VGGFeatures(layers=layers, requires_grad=False)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize to ImageNet mean/std
        mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std  = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        return (x - mean) / std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.vgg(self._norm(pred))
        t = self.vgg(self._norm(target))
        loss = 0.0
        for pi, ti in zip(p, t):
            loss = loss + F.mse_loss(pi, ti)
        return self.w * loss


# -------- Task-aware losses (Stage-B) --------
class ResponseConsistencyLoss(nn.Module):
    """Keep global brightness/contrast similar to input: mean & var penalty."""
    def __init__(self, mean_w: float = 1.0, var_w: float = 1.0):
        super().__init__()
        self.mean_w = mean_w
        self.var_w  = var_w

    def forward(self, pred: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
        # operate in grayscale
        if pred.shape[1] == 3:
            pg = 0.2989*pred[:,0:1] + 0.5870*pred[:,1:2] + 0.1140*pred[:,2:3]
            ig = 0.2989*inp[:,0:1]  + 0.5870*inp[:,1:2]  + 0.1140*inp[:,2:3]
        else:
            pg, ig = pred, inp
        m1 = pg.mean(dim=(2,3))
        m2 = ig.mean(dim=(2,3))
        v1 = pg.var(dim=(2,3), unbiased=False)
        v2 = ig.var(dim=(2,3), unbiased=False)
        return self.mean_w * F.mse_loss(m1, m2) + self.var_w * F.mse_loss(v1, v2)


class LoFTRMatchingImprovement(nn.Module):
    """
    Encourages higher LoFTR matches vs. the original pair:
    L_loftr = - (matches_enh / N - matches_orig / N)
    Minimizing L increases the improvement margin.
    """
    def __init__(self, device: str = "cuda", pretrained: str = "outdoor"):
        super().__init__()
        if not _HAS_LOFTR:
            raise RuntimeError("kornia not available for LoFTR matching loss.")
        self.device = device
        self.loftr = LoFTR(pretrained=pretrained).to(device).eval()

    @torch.no_grad()
    def _num_matches(self, img0: torch.Tensor, img1: torch.Tensor) -> torch.Tensor:
        """img*: [B,1,H,W] in [0,1] float."""
        out = self.loftr({"image0": img0, "image1": img1})
        if ("keypoints0" not in out) or (out["keypoints0"].numel() == 0):
            return img0.new_zeros(())
        return torch.tensor([out["keypoints0"].shape[0]], device=img0.device, dtype=img0.dtype).mean()

    def forward(self, pred0: torch.Tensor, pred1: torch.Tensor, inp0: torch.Tensor, inp1: torch.Tensor) -> torch.Tensor:
        # grayscale
        def to_gray(x):
            return (0.2989*x[:,0:1] + 0.5870*x[:,1:2] + 0.1140*x[:,2:3]) if x.shape[1]==3 else x
        p0 = to_gray(pred0).clamp(0,1)
        p1 = to_gray(pred1).clamp(0,1)
        i0 = to_gray(inp0).clamp(0,1)
        i1 = to_gray(inp1).clamp(0,1)

        with torch.no_grad():
            n_orig = self._num_matches(i0, i1)
        n_enh  = self._num_matches(p0, p1)

        # normalize by (H*W) to avoid scale bias; use shapes from p0
        H, W = p0.shape[-2:]
        N = float(H * W)
        # Loss is negative improvement
        loss = -(n_enh / N - n_orig / N)
        return loss
