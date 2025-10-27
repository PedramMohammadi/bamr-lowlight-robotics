# %% [markdown]
# # Stage‑A Training (paired pretraining)
# Loads configs/bamr_stage_a.yaml and runs PSNR-oriented training.
# Saves checkpoints to checkpoints/bamr_stageA and sample outputs to reports/bamr_stageA_samples.

# %%
import sys
from pathlib import Path

PROJECT_ROOT = Path("/content/drive/MyDrive/bamr_project")
CONFIG = PROJECT_ROOT / "configs" / "bamr_stage_a.yaml"

# %%
# Optional: install local package path for "bamr"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# %%
# Import our trainer
from bamr.train_stage_a import main as train_stage_a_main  # assumes I exposed a main(cfg_path)

# %%
print("Using config:", CONFIG)
assert CONFIG.exists(), f"Config not found: {CONFIG}"

# %%
# Run training
train_stage_a_main(str(CONFIG))

# %%
# After training, print the best checkpoint path if stored in config/standard location
best_ckpt = PROJECT_ROOT / "checkpoints" / "bamr_stageA" / "bamr_stageA_best.pt"
print(("✅ Found best ckpt" if best_ckpt.exists() else "❌ Missing best ckpt"), best_ckpt)
