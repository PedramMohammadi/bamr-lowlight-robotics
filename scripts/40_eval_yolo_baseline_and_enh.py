#!/usr/bin/env python3
"""
Evaluate baseline vs. BAMR-enhanced ExDark val/test and write robust CSV/JSON.

Example:
  python scripts/40_eval_yolo_baseline_and_enh.py \
    --project_root /content/drive/MyDrive/bamr_project \
    --yaml_base configs/exdark_yolo.yaml \
    --yaml_enh  configs/exdark_yolo_enh.yaml \
    --name_base exdark_baseline_v8n_py \
    --name_enh  exdark_enh_v8n_py
"""
from __future__ import annotations
import argparse
from pathlib import Path
from detector.yolo_utils import run_val, describe_dataset_yaml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default="/content/drive/MyDrive/bamr_project")
    ap.add_argument("--model", type=str, default="yolov8n.pt")
    ap.add_argument("--yaml_base", type=str, default="configs/exdark_yolo.yaml")
    ap.add_argument("--yaml_enh",  type=str, default="configs/exdark_yolo_enh.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--name_base", type=str, default="exdark_baseline_v8n_py")
    ap.add_argument("--name_enh",  type=str, default="exdark_enh_v8n_py")
    ap.add_argument("--project_out", type=str, default="reports/detector_eval")
    ap.add_argument("--half", action="store_true", default=True)
    ap.add_argument("--save_json", action="store_true", default=True)
    args = ap.parse_args()

    PR = Path(args.project_root)
    base_yaml = PR / args.yaml_base
    enh_yaml  = PR / args.yaml_enh
    out_dir   = PR / args.project_out

    print("Using:\n ", base_yaml, "\n ", enh_yaml)
    print("[base] ", describe_dataset_yaml(base_yaml))
    print("[enh ] ", describe_dataset_yaml(enh_yaml))

    md_base, r_base = run_val(
        model_path=args.model,
        data_yaml=str(base_yaml),
        out_project=str(out_dir),
        run_name=args.name_base,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        half=args.half,
        save_json=args.save_json,
        verbose=True,
    )
    md_enh, r_enh = run_val(
        model_path=args.model,
        data_yaml=str(enh_yaml),
        out_project=str(out_dir),
        run_name=args.name_enh,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        half=args.half,
        save_json=args.save_json,
        verbose=True,
    )

    print("Baseline metrics:", md_base)
    print("Enhanced metrics:", md_enh)
    if all(k in md_base and k in md_enh for k in ("mAP50","mAP50_95")):
        try:
            d50 = (md_enh["mAP50"] - md_base["mAP50"]) * 100.0
            d95 = (md_enh["mAP50_95"] - md_base["mAP50_95"]) * 100.0
            print(f"ΔmAP@50   : {d50:.2f} pts")
            print(f"ΔmAP50-95 : {d95:.2f} pts")
        except Exception:
            pass

if __name__ == "__main__":
    main()
