# %% [markdown]
# # Stage‑B Training (task-aware tuning)
# Alternates detector batches (ExDark) and MID LoFTR batches to improve downstream keypoint/detector metrics.

# %%
import sys
from pathlib import Path

PROJECT_ROOT = Path("/content/drive/MyDrive/bamr_project")
CONFIG = PROJECT_ROOT / "configs" / "bamr_stage_b.yaml"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# %%
from bamr.train_stage_b import main as train_stage_b_main  # assumes trainer main(cfg_path)

# %%
print("Using config:", CONFIG)
assert CONFIG.exists(), f"Config not found: {CONFIG}"

# %%
# Run Stage‑B training
train_stage_b_main(str(CONFIG))

# %%
# Where the Stage‑B model should be saved
ckptB = PROJECT_ROOT / "checkpoints" / "bamr_stageB" / "bamr_taskaware.pt"
print(("✅ Found Stage‑B ckpt" if ckptB.exists() else "❌ Missing Stage‑B ckpt"), ckptB)

# %%
# (Optional) Generate enhanced ExDark val/test with the trained Stage‑B model for detector eval
# I provided a CLI for this in scripts or you can call your enhancer util here if available.
# Example (CLI style):
# import subprocess
# subprocess.check_call([sys.executable, "-m", "scripts.30_train_stage_b", "--export-enhanced"])
