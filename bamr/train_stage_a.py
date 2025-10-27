# bamr/train_stage_a.py
from __future__ import annotations
import argparse
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import ConcatDataset
from typing import List
from .models import TinyBAMR
from .losses import L1Loss, EdgeLoss, PerceptualLoss
from .data import PairedTSVDataset, make_loader
from .eval_utils import eval_psnr_patches
from .utils import seed_everything, amp_autocast, save_checkpoint

def build_datasets(train_tsvs: List[str], val_tsvs: List[str], patch: int = 128):
    train_ds = ConcatDataset([PairedTSVDataset([p], patch=patch, train=True) for p in train_tsvs])
    val_ds   = ConcatDataset([PairedTSVDataset([p], patch=None, train=False) for p in val_tsvs])
    return train_ds, val_ds

def main():
    parser = argparse.ArgumentParser("Stage-A Paired Pretraining (PSNR-oriented)")
    parser.add_argument("--train_tsv", nargs="+", required=True)
    parser.add_argument("--val_tsv",   nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch",  type=int, default=8)
    parser.add_argument("--patch",  type=int, default=128)
    parser.add_argument("--lr",     type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/bamr_stageA")
    parser.add_argument("--perc_w", type=float, default=0.2)
    parser.add_argument("--edge_w", type=float, default=0.5)
    parser.add_argument("--l1_w",   type=float, default=1.0)
    parser.add_argument("--val_patches", type=int, default=512)
    args = parser.parse_args()

    seed_everything(123)
    device = args.device if torch.cuda.is_available() else "cpu"

    model = TinyBAMR(base=32).to(device)
    if device == "cuda": model.half()

    # Losses
    l1   = L1Loss()
    edge = EdgeLoss(mode="l1")
    try:
        perc = PerceptualLoss(weight=1.0)
        has_perc = True
    except Exception:
        print("[WARN] PerceptualLoss unavailable (torchvision). Skipping.")
        has_perc = False

    # Data
    train_ds, val_ds = build_datasets(args.train_tsv, args.val_tsv, patch=args.patch)
    train_loader = make_loader(train_ds, args.batch, workers=2, shuffle=True)

    opt = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs*len(train_loader))

    best_psnr = -1.0
    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "bamr_stageA_best.pt"

    step = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        for low, gt in train_loader:
            low = low.to(device)
            gt  = gt.to(device)
            if device == "cuda":
                low = low.half(); gt = gt.half()

            opt.zero_grad(set_to_none=True)
            with amp_autocast(device):
                pred = model(low)
                loss_l1   = l1(pred, gt)
                loss_edge = edge(pred, gt)
                loss = args.l1_w*loss_l1 + args.edge_w*loss_edge
                if has_perc:
                    loss_perc = perc(pred.float(), gt.float())
                    loss = loss + args.perc_w*loss_perc
                else:
                    loss_perc = torch.tensor(0.0, device=device)

            loss.backward()
            opt.step()
            scheduler.step()

            step += 1
            if step % 100 == 0:
                print(f"[{step}] loss={loss.item():.4f} l1={loss_l1.item():.4f} edge={loss_edge.item():.4f} perc={loss_perc.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}")

            if step % 1000 == 0:
                # quick val
                v_psnr = eval_psnr_patches(model, val_ds, n_patches=args.val_patches, patch=args.patch, device=device)
                print(f"  -> VAL PSNR: {v_psnr:.2f} dB on {args.val_patches} patches")
                if v_psnr > best_psnr:
                    best_psnr = v_psnr
                    save_checkpoint(str(best_path), model, extra={"best_psnr": best_psnr})

    print(f"[*] Done. Best PSNR={best_psnr:.2f} dB | saved -> {best_path}")

if __name__ == "__main__":
    main()
