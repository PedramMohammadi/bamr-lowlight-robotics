#!/usr/bin/env python3
"""
Prepare ExDark -> YOLO format (uses already-converted labels if present).
Writes dataset YAML to configs/exdark_yolo.yaml and reports split counts.

Assumes you already have prepared directories at:
  data/prepared/exdark/images/{train,val,test}
  data/prepared/exdark/labels/{train,val,test}

"""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".JPG",".JPEG",".PNG",".BMP"}

def count_files(img_dir: Path) -> tuple[int,int]:
    lbl_dir = Path(str(img_dir).replace("/images/","/labels/"))
    n_img = sum(1 for p in img_dir.rglob("*") if p.suffix in IMG_EXTS)
    n_lbl = sum(1 for p in lbl_dir.rglob("*.txt"))
    return n_img, n_lbl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default="/content/drive/MyDrive/bamr_project")
    ap.add_argument("--dataset_name", type=str, default="exdark")  # in prepared/
    ap.add_argument("--yaml_out", type=str, default=None)
    args = ap.parse_args()

    PR = Path(args.project_root)
    prep = PR / "data" / "prepared" / args.dataset_name
    imgs_train = prep / "images" / "train"
    imgs_val   = prep / "images" / "val"
    imgs_test  = prep / "images" / "test"

    if not imgs_train.exists() or not imgs_val.exists() or not imgs_test.exists():
        print("[ERROR] Prepared ExDark folders not found:")
        print("  ", imgs_train)
        print("  ", imgs_val)
        print("  ", imgs_test)
        print("Please re-run your Step 4 label rebuild & conversion, then retry.")
        return

    ntr = count_files(imgs_train)
    nva = count_files(imgs_val)
    nte = count_files(imgs_test)
    print("[Counts]")
    print("train images/labels:", ntr)
    print("val   images/labels:", nva)
    print("test  images/labels:", nte)

    cfg_dir = PR / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = Path(args.yaml_out) if args.yaml_out else cfg_dir / "exdark_yolo.yaml"

    y = {
        "path": "",
        "train": str(imgs_train),
        "val":   str(imgs_val),
        "test":  str(imgs_test),
        "nc": 12,
        "names": [
            "Bicycle","Boat","Bottle","Bus","Car","Cat",
            "Chair","Cup","Dog","Motorbike","People","Table"
        ]
    }
    with open(yaml_path, "w") as f:
        yaml.safe_dump(y, f, sort_keys=False)

    print("[Dataset yaml]")
    print(yaml_path.read_text())

if __name__ == "__main__":
    main()
