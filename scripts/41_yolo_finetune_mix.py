#!/usr/bin/env python3
"""
Short YOLO fine-tune on original/enhanced (merged or sequential), then validate.

You can pass multiple YAMLs to merge their TRAIN dirs:
  python scripts/41_yolo_finetune_mix.py \
    --project_root /content/drive/MyDrive/bamr_project \
    --model yolov8n.pt \
    --data configs/exdark_yolo.yaml \
    --data_extra configs/exdark_yolo_enh.yaml \
    --epochs 10 --batch 16 --name ft_mixed

If your local Ultralytics build doesn’t support list-of-train-dirs,
run sequentially twice (first on original, then on enhanced).
"""
from __future__ import annotations
from detector.yolo_finetune import main as ft_main

# Thin proxy so your repo has a stable CLI name.
# (Actual logic resides in detector/yolo_finetune.py)

def main():
    # Forward argparse to ft_main via a small wrapper
    ft_main()

if __name__ == "__main__":
    main()
