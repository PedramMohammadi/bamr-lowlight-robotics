# detector/yolo_val.py
from __future__ import annotations

import argparse
from .yolo_utils import describe_dataset_yaml, run_val

def main():
    ap = argparse.ArgumentParser("YOLO validation with robust CSV/JSON export")
    ap.add_argument("--model", type=str, default="yolov8n.pt", help="Path to .pt (or hub name)")
    ap.add_argument("--data", type=str, required=True, help="Dataset YAML")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--half", action="store_true", default=True)
    ap.add_argument("--name", type=str, default="eval_run")
    ap.add_argument("--project", type=str, default="runs/detector_eval")
    ap.add_argument("--save_json", action="store_true", default=True)
    args = ap.parse_args()

    # Print a quick dataset sanity line
    info = describe_dataset_yaml(args.data)
    print("Dataset:", info)

    md, run_dir = run_val(
        model_path=args.model,
        data_yaml=args.data,
        out_project=args.project,
        run_name=args.name,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        half=args.half,
        save_json=args.save_json,
        verbose=True,
    )

    print("Metrics:", md)
    print("Run dir:", run_dir)

if __name__ == "__main__":
    main()
