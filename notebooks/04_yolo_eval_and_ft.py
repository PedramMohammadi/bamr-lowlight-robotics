# %% [markdown]
# # YOLO: Evaluate & Short Fine‑Tune
# - Evaluate YOLOv8n on original ExDark val/test
# - Evaluate on BAMR‑enhanced val/test
# - (Optional) short fine‑tune on original vs enhanced vs mixed
# - Write results.csv + predictions.json and print deltas

# %%
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path("/content/drive/MyDrive/bamr_project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

ORIG_YAML = PROJECT_ROOT / "configs" / "exdark_yolo.yaml"
ENH_YAML  = PROJECT_ROOT / "configs" / "exdark_yolo_enh.yaml"
OUT_DIR   = PROJECT_ROOT / "reports" / "detector_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Configs:")
print("  ", ORIG_YAML)
print("  ", ENH_YAML)

# %%
from ultralytics import YOLO

def yolo_val_to_csv(data_yaml: Path, run_name: str, imgsz=640, batch=8, device=0, half=True) -> Path:
    model = YOLO("yolov8n.pt")
    metrics = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=batch,
        device=device,
        half=half,
        save_json=True,
        project=str(OUT_DIR),
        name=run_name,
        verbose=True,
    )
    run_dir = OUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "results.csv"
    try:
        csv_text = metrics.to_csv()
        csv_path.write_text(csv_text)
    except Exception:
        md = getattr(metrics, "results_dict", {})
        pd.DataFrame([md]).to_csv(csv_path, index=False)
    print("Wrote:", csv_path)
    return csv_path

def read_maps(csv_path: Path):
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    def grab(*keys):
        for k in keys:
            k_lower = k.lower()
            if k_lower in cols:
                return float(df.iloc[-1][cols[k_lower]])
        return float("nan")
    return {
        "mAP50":     grab("map50","metrics/map50","map_0.5","mAP@0.5"),
        "mAP50_95":  grab("map50-95","map_50_95","metrics/mAP50-95","mAP@0.5:0.95"),
        "precision": grab("precision","metrics/precision"),
        "recall":    grab("recall","metrics/recall"),
    }

# %%
# Baseline (original val)
base_csv = yolo_val_to_csv(ORIG_YAML, "exdark_baseline_v8n_py")
base = read_maps(base_csv)
print("Baseline:", base)

# Enhanced (enhanced val)
enh_csv = yolo_val_to_csv(ENH_YAML, "exdark_enh_v8n_py")
enh = read_maps(enh_csv)
print("Enhanced:", enh)

# %%
def pct(x): 
    return "nan" if (x!=x) else f"{x*100:.2f} pts"

if pd.notna(base["mAP50"]) and pd.notna(enh["mAP50"]):
    print("Δ mAP@0.5   :", pct(enh["mAP50"] - base["mAP50"]))
if pd.notna(base["mAP50_95"]) and pd.notna(enh["mAP50_95"]):
    print("Δ mAP@0.5:0.95:", pct(enh["mAP50_95"] - base["mAP50_95"]))

# %%
# (Optional) quick fine‑tune examples:
# - On original train, validate on original val
# - On original train, validate on enhanced val
# - On original+enhanced mix (requires you to create a mixed YAML if desired)

# Example: short FT, 3 epochs, validate on enhanced val after
# model = YOLO("yolov8n.pt")
# model.train(data=str(ORIG_YAML), epochs=3, imgsz=640, batch=16, device=0, workers=2, name="exdark_ft3_orig_train")
# model.val(data=str(ENH_YAML), imgsz=640, batch=8, device=0, name="exdark_ft3_origtrain_on_enh_val")
