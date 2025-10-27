#!/usr/bin/env python3
"""
Build LOL-v2 Real paired TSVs using the fixed layout you confirmed:

  data/raw/lolv2_real/LOL-v2/Real_captured/Train/{Low,Normal}
  data/raw/lolv2_real/LOL-v2/Real_captured/Test/{Low,Normal}

Writes:
  data/lists/lolv2_train.tsv
  data/lists/lolv2_val.tsv
  data/lists/lolv2_test.tsv
(split 90/10 within Train; Test goes to test TSV)

Row format: <low_path>\t<gt_path>
"""
from __future__ import annotations
import argparse, random
from pathlib import Path

IMG_EXTS = {".png",".jpg",".jpeg",".JPG",".JPEG",".PNG"}

def list_images(d: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(p for p in d.rglob("*") if p.suffix in IMG_EXTS)}

def write_tsv(pairs, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for a,b in pairs:
            f.write(f"{a}\t{b}\n")
    print(f"Wrote {len(pairs)} -> {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default="/content/drive/MyDrive/bamr_project")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.10)  # of TRAIN
    args = ap.parse_args()

    random.seed(args.seed)
    PR = Path(args.project_root)
    rc = PR / "data" / "raw" / "lolv2_real" / "LOL-v2" / "Real_captured"
    low_tr = rc / "Train" / "Low"
    gt_tr  = rc / "Train" / "Normal"
    low_te = rc / "Test" / "Low"
    gt_te  = rc / "Test" / "Normal"

    for d in [low_tr, gt_tr, low_te, gt_te]:
        if not d.exists():
            print("[ERROR] Missing:", d)
            return

    Ltr = list_images(low_tr)
    Gtr = list_images(gt_tr)
    Lte = list_images(low_te)
    Gte = list_images(gt_te)

    stems_tr = sorted(set(Ltr).intersection(Gtr))
    stems_te = sorted(set(Lte).intersection(Gte))

    pairs_tr = [(Ltr[s], Gtr[s]) for s in stems_tr]
    pairs_te = [(Lte[s], Gte[s]) for s in stems_te]
    random.shuffle(pairs_tr)

    n = len(pairs_tr)
    n_val = int(round(n * args.val_frac))
    n_train = n - n_val
    train = pairs_tr[:n_train]
    val   = pairs_tr[n_train:]

    out_dir = PR / "data" / "lists"
    write_tsv(train, out_dir / "lolv2_train.tsv")
    write_tsv(val,   out_dir / "lolv2_val.tsv")
    write_tsv(pairs_te, out_dir / "lolv2_test.tsv")

if __name__ == "__main__":
    main()
