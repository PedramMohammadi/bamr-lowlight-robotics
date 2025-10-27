# %% [markdown]
# # Prepare Datasets (sanity & on-demand)
# This notebook verifies prepared splits and (optionally) calls the CLI scripts:
# - scripts/10_prepare_exdark.py
# - scripts/11_prepare_lol_blur.py
# - scripts/12_prepare_lolv2.py
# - scripts/13_prepare_mid_mini.py

# %%
import sys, subprocess
from pathlib import Path

PROJECT_ROOT = Path("/content/drive/MyDrive/bamr_project")
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"

def run_cli(module_name, *args):
    """Run one of our CLI scripts, e.g., python -m scripts.10_prepare_exdark --arg ..."""
    cmd = [sys.executable, "-m", f"scripts.{module_name}", *map(str, args)]
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)

# %%
# 0) Verify ExDark YOLO dirs exist (train/val/test images + labels)
def check_exdark_prepared():
    base = PROJECT_ROOT / "data" / "prepared" / "exdark"
    sub = ["images/train", "labels/train", "images/val", "labels/val", "images/test", "labels/test"]
    ok = True
    for s in sub:
        p = base / s
        ok &= p.exists()
        print(("✅" if p.exists() else "❌"), p)
    return ok

exdark_ok = check_exdark_prepared()
if not exdark_ok:
    # If you need to build from raw ExDark annotations, run:
    # run_cli("10_prepare_exdark")
    print("ExDark prepared set missing. If you have the raw ExDark, run the prep script above.")

# %%
# 1) Verify paired TSVs for LOL-Blur and LOLv2 (created in your Step 3)
for name in ["lol_blur_train.tsv", "lol_blur_val.tsv", "lol_blur_test.tsv",
             "lolv2_train.tsv", "lolv2_val.tsv", "lolv2_test.tsv"]:
    p = PROJECT_ROOT / "data" / "lists" / name
    print(("✅" if p.exists() else "❌"), p)

# If not present, you can generate them:
# run_cli("11_prepare_lol_blur")
# run_cli("12_prepare_lolv2")

# %%
# 2) Verify MID-mini prepared images (viewA/viewB JPGs)
mid_img_root = PROJECT_ROOT / "data" / "prepared" / "mid_mini" / "images"
print(("✅" if mid_img_root.exists() else "❌"), mid_img_root)
# If missing but you placed CR2/JPG pairs in raw/..., run:
# run_cli("13_prepare_mid_mini")

# %%
print("Dataset prep sanity check complete.")
