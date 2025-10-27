# bamr/eval_utils.py
from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from .utils import psnr, amp_autocast
import numpy as np

@torch.no_grad()
def eval_psnr_patches(
    model: nn.Module,
    dataset: Dataset,
    n_patches: int = 512,
    patch: int = 128,
    device: str = "cuda"
) -> float:
    """
    Draws random patches from the dataset and computes mean PSNR.
    Expects dataset to return (low, gt) in [0,1].
    """
    model.eval()
    total = 0.0
    count = 0
    rng = np.random.default_rng(123)
    idxs = rng.integers(0, len(dataset), size=n_patches)
    for i in idxs:
        low, gt = dataset[i]
        H, W = low.shape[-2:]
        if H < patch or W < patch:
            # pad/crop naive
            pad_h = max(0, patch - H); pad_w = max(0, patch - W)
            low = torch.nn.functional.pad(low, (0,pad_w,0,pad_h), mode="replicate")
            gt  = torch.nn.functional.pad(gt,  (0,pad_w,0,pad_h), mode="replicate")
            H, W = low.shape[-2:]

        y = rng.integers(0, H - patch + 1); x = rng.integers(0, W - patch + 1)
        low_p = low[:, y:y+patch, x:x+patch].unsqueeze(0).to(device)
        gt_p  = gt[:,  y:y+patch, x:x+patch].unsqueeze(0).to(device)

        with amp_autocast(device):
            pred = model(low_p)
        total += float(psnr(pred, gt_p).cpu())
        count += 1
    return total / max(1, count)
