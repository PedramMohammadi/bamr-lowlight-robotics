# detector/yolo_utils.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import yaml
import pandas as pd

from ultralytics import YOLO


IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".JPG",".JPEG",".PNG",".BMP"}


# ---------------------------
# YAML / dataset sanity checks
# ---------------------------
def _load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _count_images_labels(images_dir: str | Path) -> Tuple[int, int]:
    images_dir = Path(images_dir)
    labels_dir = Path(str(images_dir).replace("/images/", "/labels/"))
    n_imgs = sum(1 for p in images_dir.rglob("*") if p.suffix in IMG_EXTS)
    n_lbls = sum(1 for p in labels_dir.rglob("*.txt"))
    return n_imgs, n_lbls


def describe_dataset_yaml(yaml_path: str | Path) -> Dict[str, Any]:
    """Return counts and resolved paths for train/val/test (if present)."""
    y = _load_yaml(yaml_path)
    out = {"yaml": str(yaml_path)}
    for split in ("train", "val", "test"):
        if split in y and y[split]:
            path = y[split]
            out[f"{split}_path"] = path
            try:
                n_img, n_lbl = _count_images_labels(path)
            except Exception:
                n_img, n_lbl = 0, 0
            out[f"{split}_images"] = n_img
            out[f"{split}_labels"] = n_lbl
    return out


# ---------------------------
# Val + CSV/JSON persistence
# ---------------------------
def _write_results_csv_from_metrics(metrics, csv_path: Path) -> None:
    """
    Ultralytics >= 8.3 exposes metrics.to_csv(); older versions carry results_dict.
    I try both to ensure 'results.csv' always exists.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Newer API
        csv_text = metrics.to_csv()  # type: ignore[attr-defined]
        csv_path.write_text(csv_text)
    except Exception:
        md = getattr(metrics, "results_dict", None)
        if md is None:
            # As a last resort, try to derive from attributes
            md = {}
            for k in ("box", "speed"):
                if hasattr(metrics, k):
                    try:
                        md[k] = getattr(metrics, k)
                    except Exception:
                        pass
        df = pd.DataFrame([md]) if isinstance(md, dict) else pd.DataFrame(md)
        df.to_csv(csv_path, index=False)


def _metrics_dict(metrics) -> Dict[str, float]:
    """
    Normalize common fields regardless of ultralytics minor version.
    """
    rd = getattr(metrics, "results_dict", {}) or {}
    # Try common keys, fallback gracefully
    def g(*keys):
        for k in keys:
            if k in rd:
                return rd[k]
        return float("nan")
    return {
        "mAP50":     g("metrics/mAP50", "map50", "map_50", "map@50"),
        "mAP50_95":  g("metrics/mAP50-95", "map50-95", "map_50_95", "map"),
        "precision": g("metrics/precision", "precision"),
        "recall":    g("metrics/recall", "recall"),
    }


def run_val(
    model_path: str | Path,
    data_yaml: str | Path,
    out_project: str | Path,
    run_name: str,
    imgsz: int = 640,
    batch: int = 8,
    device: int | str = 0,
    half: bool = True,
    save_json: bool = True,
    verbose: bool = True,
) -> Tuple[Dict[str, float], Path]:
    """
    Validate a YOLO model on a dataset YAML.
    Writes predictions.json and results.csv into the run folder.
    Returns (metrics_dict, run_dir).
    """
    out_project = Path(out_project)
    out_project.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=batch,
        device=device,
        half=half,
        save_json=save_json,
        project=str(out_project),
        name=run_name,
        verbose=verbose,
    )

    run_dir = out_project / run_name
    # Ensure predictions.json (ultralytics writes automatically when save_json=True)
    # Ensure results.csv
    _write_results_csv_from_metrics(metrics, run_dir / "results.csv")

    # Summary dict for quick comparisons
    md = _metrics_dict(metrics)

    # Echo for logs
    if verbose:
        print(f"Results saved to {run_dir}")
        print("Summary:", md)

    return md, run_dir


# ---------------------------
# Optional helpers
# ---------------------------
def read_results_csv(csv_path: str | Path) -> Dict[str, float]:
    df = pd.read_csv(csv_path)
    # Normalize header names to safer keys
    cols = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols:
                return float(df.iloc[-1][cols[n]])
        return float("nan")
    return {
        "mAP50": pick("map50", "metrics/map50"),
        "mAP50_95": pick("map50-95", "metrics/map50-95", "map_50_95"),
        "precision": pick("precision", "metrics/precision"),
        "recall": pick("recall", "metrics/recall"),
    }


def ensure_predictions_json(run_dir: str | Path) -> Optional[Path]:
    run_dir = Path(run_dir)
    pj = run_dir / "predictions.json"
    return pj if pj.exists() else None
