#!/usr/bin/env python3
"""
Stage‑B: task-aware tuning with detector features + LoFTR/RC

This script calls into bamr.train_stage_b which implements the alternating
detector / MID batches and your robust AMP logic.

Example:
  python scripts/30_train_stage_b.py --project_root /content/drive/MyDrive/bamr_project \
    --stageA_ckpt checkpoints/bamr_stageA/bamr_stageA_best.pt \
    --out_ckpt checkpoints/bamr_stageB/bamr_taskaware.pt
"""
from __future__ import annotations
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default="/content/drive/MyDrive/bamr_project")
    ap.add_argument("--stageA_ckpt", type=str, default="checkpoints/bamr_stageA/bamr_stageA_best.pt")
    ap.add_argument("--mid_img_root", type=str, default="data/prepared/mid_mini/images")
    ap.add_argument("--mid_raw_root", type=str, default="data/raw/mid/mini")
    ap.add_argument("--epochs", type=int, default=7500)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--det_yaml", type=str, default="configs/exdark_yolo.yaml")
    ap.add_argument("--out_ckpt", type=str, default="checkpoints/bamr_stageB/bamr_taskaware.pt")
    ap.add_argument("--prior_w", type=float, default=0.20)
    ap.add_argument("--rc_w", type=float, default=0.05)
    ap.add_argument("--loftr_w", type=float, default=0.75)
    args = ap.parse_args()

    PR = Path(args.project_root)
    from bamr.train_stage_b import train_stage_b_main  # you’ll provide this in bamr/

    train_stage_b_main(
        project_root=PR,
        stageA_ckpt=PR / args.stageA_ckpt,
        mid_img_root=PR / args.mid_img_root,
        mid_raw_root=PR / args.mid_raw_root,
        det_yaml=PR / args.det_yaml,
        max_steps=args.epochs,
        batch=args.batch,
        prior_w=args.prior_w,
        rc_w=args.rc_w,
        loftr_w=args.loftr_w,
        out_ckpt=PR / args.out_ckpt,
    )

if __name__ == "__main__":
    main()
