#!/usr/bin/env python3
"""
Stage‑A: paired pretraining (PSNR/SSIM-oriented) on LOL-Blur + LOL-v2 Real lists.

Relies on 'bamr' package modules you asked for (models/losses/data/train_stage_a).
If you haven't added bamr/ yet, place it at repo root (importable as 'bamr').

Usage example:
  python scripts/20_train_stage_a.py --project_root /content/drive/MyDrive/bamr_project \
    --lol_blur_tsv data/lists/lol_blur_train.tsv --lolv2_tsv data/lists/lolv2_train.tsv
"""
from __future__ import annotations
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default="/content/drive/MyDrive/bamr_project")
    ap.add_argument("--lol_blur_tsv", type=str, default="data/lists/lol_blur_train.tsv")
    ap.add_argument("--lolv2_tsv", type=str, default="data/lists/lolv2_train.tsv")
    ap.add_argument("--val_patches", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=150000)  # your step-based loop count
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints/bamr_stageA")
    args = ap.parse_args()

    PR = Path(args.project_root)
    from bamr.train_stage_a import train_stage_a_main  # you’ll provide this in bamr/

    train_stage_a_main(
        project_root=PR,
        lol_blur_tsv=PR / args.lol_blur_tsv,
        lolv2_tsv=PR / args.lolv2_tsv,
        val_patches=args.val_patches,
        max_steps=args.epochs,
        batch=args.batch,
        lr=args.lr,
        ckpt_dir=PR / args.ckpt_dir,
    )

if __name__ == "__main__":
    main()
