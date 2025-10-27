#!/usr/bin/env python3
"""
Build LOL-Blur paired file lists (train/val/test) as TSVs from raw directory:

Expected layout (from your drive):
  data/raw/lol_blur/train/low_blur
  data/raw/lol_blur/train/low_blur_noise     (optional, included if present)
  data/raw/lol_blur/train/high_sharp_original
  data/raw/lol_blur/test/test

Writes:
  data/lists/lol_blur_train.tsv
  data/lists/lol_blur_val.tsv
  data/lists/lol_blur_test.tsv

Each TSV row: <low_path>\t<gt_path>
"""
from __future__ import annotations
import argparse, random
from pathlib import Path

IMG_EXTS = {".png",".jpg",".jpeg",".JPG",".JPEG",".PNG"}

def list_images(d: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(p for p in d.rglob("*") if p.suffix in IMG_EXTS)}

def write_tsv(pairs: list[tuple[Path,Path]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for lp, gp in pairs:
            f.write(f"{lp}\t{gp}\n")
    print(f"Wrote {len(pairs)} -> {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default="/content/drive/MyDrive/bamr_project")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--test_frac", type=float, default=0.10)
    args = ap.parse_args()

    random.seed(args.seed)
    PR = Path(args.project_root)
    raw = PR / "data" / "raw" / "lol_blur" / "train"
    test_raw = PR / "data" / "raw" / "lol_blur" / "test"  # may be "test/test"; handle both
    if (test_raw / "test").exists():
        test_raw = test_raw / "test"

    low_dirs = [raw / "low_blur"]
    if (raw / "low_blur_noise").exists():
        low_dirs.append(raw / "low_blur_noise")
    gt_dir = raw / "high_sharp_original"

    for d in low_dirs + [gt_dir, test_raw]:
        if not d.exists():
            print("[ERROR] Missing:", d)
            return

    # Build low→gt pairs by common stem
    gt_map = list_images(gt_dir)
    low_maps = [list_images(d) for d in low_dirs]
    low_all = {}
    for lm in low_maps:
        low_all.update(lm)

    stems = sorted(set(low_all).intersection(gt_map))
    pairs = [(low_all[s], gt_map[s]) for s in stems]
    random.shuffle(pairs)

    n = len(pairs)
    n_val = int(round(n * args.val_frac))
    n_test = int(round(n * args.test_frac))
    n_train = n - n_val - n_test
    train = pairs[:n_train]
    val   = pairs[n_train:n_train+n_val]
    test  = pairs[n_train+n_val:]

    out_dir = PR / "data" / "lists"
    write_tsv(train, out_dir / "lol_blur_train.tsv")
    write_tsv(val,   out_dir / "lol_blur_val.tsv")
    write_tsv(test,  out_dir / "lol_blur_test.tsv")

if __name__ == "__main__":
    main()
