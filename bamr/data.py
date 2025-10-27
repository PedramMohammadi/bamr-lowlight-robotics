# bamr/data.py
from __future__ import annotations
import random, yaml
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".JPG",".JPEG",".PNG",".BMP"}

def _imread_rgb(path: str) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def _to_tensor_u8_to_f32(x: np.ndarray) -> torch.Tensor:
    # [H,W,3] uint8 -> [3,H,W] float32 [0,1]
    t = torch.from_numpy(x).permute(2,0,1).contiguous()
    return t.float() / 255.0

# ---------- Paired TSV (LOL‑Blur / LOL‑v2) ----------
class PairedTSVDataset(Dataset):
    """
    Lines: 'low_path<TAB>gt_path'
    Optional random crop/flip for training.
    """
    def __init__(self, tsv_files: List[str], patch: Optional[int] = None, train: bool = True):
        self.pairs: List[Tuple[str,str]] = []
        for tsv in tsv_files:
            with open(tsv, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    low, gt = line.split("\t")
                    self.pairs.append((low, gt))
        self.patch = patch
        self.train = train

    def __len__(self) -> int:
        return len(self.pairs)

    def _crop(self, a: np.ndarray, b: np.ndarray, s: int) -> Tuple[np.ndarray,np.ndarray]:
        H, W = a.shape[:2]
        if H < s or W < s:
            scale = max(s/H, s/W)
            nh, nw = int(H*scale+0.5), int(W*scale+0.5)
            a = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_AREA)
            b = cv2.resize(b, (nw, nh), interpolation=cv2.INTER_AREA)
            H, W = a.shape[:2]
        y = random.randint(0, H - s)
        x = random.randint(0, W - s)
        return a[y:y+s, x:x+s], b[y:y+s, x:x+s]

    def __getitem__(self, idx: int):
        low_path, gt_path = self.pairs[idx]
        lo = _imread_rgb(low_path)
        gt = _imread_rgb(gt_path)
        if self.patch is not None:
            lo, gt = self._crop(lo, gt, self.patch)
        if self.train and random.random() < 0.5:
            lo, gt = np.ascontiguousarray(lo[:, ::-1]), np.ascontiguousarray(gt[:, ::-1])
        return _to_tensor_u8_to_f32(lo), _to_tensor_u8_to_f32(gt)

# ---------- Detection images from YAML (ExDark) ----------
def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _collect_images(dir_path: str) -> List[str]:
    p = Path(dir_path)
    files = [str(pp) for pp in p.rglob("*") if pp.suffix in IMG_EXTS]
    files.sort()
    return files

class YOLOImagesDataset(Dataset):
    """Just returns images (no labels needed for Stage‑B priors)."""
    def __init__(self, yaml_path: str, split: str = "train", patch: Optional[int] = None):
        y = _load_yaml(yaml_path)
        key = {"train":"train","val":"val","test":"test"}[split]
        imgs_dir = y[key]
        self.paths = _collect_images(imgs_dir)
        self.patch = patch

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = _imread_rgb(self.paths[idx])
        if self.patch:
            # random crop for uniform compute
            H, W = img.shape[:2]; s = self.patch
            if H < s or W < s:
                sc = max(s/H, s/W); img = cv2.resize(img, (int(W*sc+0.5), int(H*sc+0.5)), interpolation=cv2.INTER_AREA)
                H, W = img.shape[:2]
            y = random.randint(0, H - s); x = random.randint(0, W - s)
            img = img[y:y+s, x:x+s]
        return _to_tensor_u8_to_f32(img)

# ---------- MID dual view pairs ----------
def _match_by_stem(A: List[Path], B: List[Path]) -> List[Tuple[str,str]]:
    A_map = {p.stem: p for p in A}
    B_map = {p.stem: p for p in B}
    commons = sorted(set(A_map).intersection(B_map))
    if commons:
        return [(str(A_map[s]), str(B_map[s])) for s in commons]
    # fallback pairwise order
    n = min(len(A), len(B))
    return [(str(A[i]), str(B[i])) for i in range(n)]

class MIDDualViewDataset(Dataset):
    """
    Returns two views (A,B) for LoFTR supervision.
    Directory layout: root/images/{Indoor|Outdoor}/pairXX/{viewA|viewB}/*.jpg
    """
    def __init__(self, images_root: str, K_per_pair: int = 3, patch: Optional[int] = None):
        self.items: List[Tuple[str,str]] = []
        root = Path(images_root)
        for domain in ["Indoor", "Outdoor"]:
            for pair_dir in sorted((root/domain).glob("pair*")):
                vA = pair_dir/"viewA"; vB = pair_dir/"viewB"
                A = sorted([p for p in vA.glob("*") if p.suffix in IMG_EXTS])
                B = sorted([p for p in vB.glob("*") if p.suffix in IMG_EXTS])
                if not A or not B: continue
                pairs = _match_by_stem(A, B)
                if len(pairs) <= K_per_pair:
                    chosen = pairs
                else:
                    idxs = [0, len(pairs)//2, -1] if K_per_pair == 3 else np.linspace(0, len(pairs)-1, K_per_pair).astype(int).tolist()  # type: ignore
                    chosen = [pairs[i] for i in idxs]
                self.items.extend(chosen)
        self.patch = patch

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pa, pb = self.items[idx]
        A = _imread_rgb(pa); B = _imread_rgb(pb)
        if self.patch:
            H, W = A.shape[:2]; s = self.patch
            if H < s or W < s:
                sc = max(s/H, s/W)
                A = cv2.resize(A, (int(W*sc+0.5), int(H*sc+0.5)), interpolation=cv2.INTER_AREA)
                B = cv2.resize(B, (int(W*sc+0.5), int(H*sc+0.5)), interpolation=cv2.INTER_AREA)
                H, W = A.shape[:2]
            y = random.randint(0, H - self.patch); x = random.randint(0, W - self.patch)
            A = A[y:y+self.patch, x:x+self.patch]; B = B[y:y+self.patch, x:x+self.patch]
        return _to_tensor_u8_to_f32(A), _to_tensor_u8_to_f32(B)

# ---------- helpers ----------
def make_loader(dataset: Dataset, batch: int, workers: int = 2, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=workers, pin_memory=True, drop_last=True)
