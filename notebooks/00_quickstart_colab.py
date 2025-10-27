# %% [markdown]
# # BAMR: Quickstart (Colab)
# - Mount Google Drive
# - Create project folders
# - Check GPU & key paths
# - Verify configs exist

# %%
import os, json
from pathlib import Path

# Improves CUDA memory behavior on Colab
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# %%
# Mount Drive if on Colab
try:
    from google.colab import drive  # type: ignore
    IN_COLAB = True
except Exception:
    IN_COLAB = False

if IN_COLAB:
    drive.mount("/content/drive", force_remount=True)

# %%
PROJECT_ROOT = Path("/content/drive/MyDrive/bamr_project")
CONFIGS      = PROJECT_ROOT / "configs"

# Create minimal structure if missing
for p in [
    PROJECT_ROOT / "data/raw",
    PROJECT_ROOT / "data/prepared",
    PROJECT_ROOT / "data/lists",
    PROJECT_ROOT / "reports",
    PROJECT_ROOT / "checkpoints",
    PROJECT_ROOT / "configs",
]:
    p.mkdir(parents=True, exist_ok=True)

print("PROJECT_ROOT:", PROJECT_ROOT)

# %%
# Check GPU
try:
    import torch
    print("Torch:", torch.__version__)
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
    else:
        print("⚠️ CUDA not available")
except Exception as e:
    print("Torch not installed yet:", e)

# %%
# Verify expected config files exist
expected = [
    CONFIGS / "bamr_stage_a.yaml",
    CONFIGS / "bamr_stage_b.yaml",
    CONFIGS / "exdark_yolo.yaml",
    CONFIGS / "exdark_yolo_enh.yaml",
]
for f in expected:
    print(("✅" if f.exists() else "❌"), f)

# %%
# Quick path summary
summ = {
    "raw": str(PROJECT_ROOT / "data/raw"),
    "prepared": str(PROJECT_ROOT / "data/prepared"),
    "lists": str(PROJECT_ROOT / "data/lists"),
    "reports": str(PROJECT_ROOT / "reports"),
    "checkpoints": str(PROJECT_ROOT / "checkpoints"),
}
print(json.dumps(summ, indent=2))
