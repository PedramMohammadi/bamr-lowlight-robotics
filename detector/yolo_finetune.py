# detector/yolo_finetune.py
from __future__ import annotations

import argparse, shutil, tempfile, yaml
from pathlib import Path
from typing import Dict, Any, List, Optional

from ultralytics import YOLO

from .yolo_utils import describe_dataset_yaml, run_val


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _dump_yaml(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _merge_train_sources(yamls: List[str | Path], out_path: str | Path) -> str:
    """
    Create a temporary merged YAML whose 'train' is a list of image dirs from multiple YAMLs.
    'val' and 'test' are taken from the FIRST YAML unless overridden later.
    NOTE: Ultralytics supports lists for 'train' since v8; if your version doesn't,
    prefer doing two short fine-tunes sequentially instead of merging.
    """
    yamls = [Path(p) for p in yamls]
    bases = [_load_yaml(p) for p in yamls]

    merged: Dict[str, Any] = {}
    merged["path"] = ""  # not used
    # Combine 'train' entries as a list
    trains: List[str] = []
    for b in bases:
        tr = b.get("train", None)
        if isinstance(tr, list):
            trains.extend(tr)
        elif isinstance(tr, str):
            trains.append(tr)
    if not trains:
        raise ValueError("No 'train' entries found to merge.")
    merged["train"] = trains

    # Borrow val/test from first YAML if present
    for k in ("val", "test", "nc", "names"):
        if k in bases[0]:
            merged[k] = bases[0][k]

    _dump_yaml(merged, out_path)
    return str(out_path)


def _find_best_ckpt(train_run_dir: Path) -> Optional[Path]:
    w = train_run_dir / "weights" / "best.pt"
    if w.exists():
        return w
    # fallback: last.pt
    w = train_run_dir / "weights" / "last.pt"
    return w if w.exists() else None


def main():
    ap = argparse.ArgumentParser("Short YOLO fine-tune on original/enhanced/mixed")
    ap.add_argument("--model", type=str, default="yolov8n.pt", help="base model")
    ap.add_argument("--data", type=str, required=True, help="primary dataset YAML (train/val)")
    ap.add_argument("--data_extra", type=str, nargs="*", default=[], help="optional: extra YAMLs to MERGE into train (val/test remain from --data)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--lr0", type=float, default=5e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", type=str, default="yolo_ft")
    ap.add_argument("--project", type=str, default="runs/detector_ft")
    ap.add_argument("--val_project", type=str, default="runs/detector_eval_ft")
    ap.add_argument("--half", action="store_true", default=True)
    ap.add_argument("--save_json", action="store_true", default=True)
    ap.add_argument("--freeze", type=int, default=0, help="freeze n first layers (0=no freeze)")
    args = ap.parse_args()

    # Build (maybe) merged training YAML
    merge_tmp_dir = Path(tempfile.mkdtemp(prefix="yolo_merge_"))
    merged_yaml = None
    if args.data_extra:
        merged_yaml = _merge_train_sources(
            [args.data] + args.data_extra,
            out_path=merge_tmp_dir / "merged.yaml"
        )
        train_data_yaml = merged_yaml
        print(f"[INFO] merged train YAML -> {train_data_yaml}")
    else:
        train_data_yaml = args.data

    # For visibility
    print("Primary dataset:", describe_dataset_yaml(args.data))
    if args.data_extra:
        for i, y in enumerate(args.data_extra, 1):
            print(f"Extra[{i}] dataset:", describe_dataset_yaml(y))

    # Train
    model = YOLO(args.model)
    train_results = model.train(
        data=train_data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        lr0=args.lr0,
        patience=args.patience,
        seed=args.seed,
        name=args.name,
        project=args.project,
        freeze=args.freeze,
        verbose=True,
        # You can uncomment these if you want:
        # cache=True,
        # cos_lr=True,
        # amp=True,
    )
    train_dir = Path(train_results.save_dir) if hasattr(train_results, "save_dir") else Path(args.project) / args.name
    print("Train dir:", train_dir)

    # Resolve best checkpoint
    best = _find_best_ckpt(train_dir)
    if best is None:
        raise FileNotFoundError(f"Could not find best.pt/last.pt under {train_dir/'weights'}")

    # Validate on the same val as primary YAML
    print("\n[VAL] On primary val set:")
    md_primary, run_primary = run_val(
        model_path=str(best),
        data_yaml=args.data,
        out_project=args.val_project,
        run_name=f"{args.name}_on_primary",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        half=args.half,
        save_json=args.save_json,
        verbose=True,
    )

    # If you want, also validate on any additional YAMLs' val sets
    for i, y in enumerate(args.data_extra, 1):
        print(f"\n[VAL] On extra[{i}] val set:")
        md_extra, run_extra = run_val(
            model_path=str(best),
            data_yaml=y,
            out_project=args.val_project,
            run_name=f"{args.name}_on_extra{i}",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            half=args.half,
            save_json=args.save_json,
            verbose=True,
        )

    # Cleanup temp
    try:
        shutil.rmtree(merge_tmp_dir)
    except Exception:
        pass

    print("\nDone.")

if __name__ == "__main__":
    main()
