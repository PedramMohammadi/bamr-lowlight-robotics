#!/usr/bin/env python3
"""
Prepare a small MID subset you already copied to Drive.

Expected input (what you built):
  data/raw/mid/mini/{Indoor,Outdoor}/pairXX/{viewA,viewB}/*.jpg
  data/raw/mid/mini/{Indoor,Outdoor}/pairXX/GT_Correspondence/E_estimated.npy  (optional but used by eval)

This script:
  1) Optionally converts CR2->JPG if you pass --convert_raw (requires rawpy).
  2) Copies (or symlinks) the existing JPGs into:
       data/prepared/mid_mini/images/{Indoor|Outdoor}/pairXX/{viewA|viewB}/
  3) Writes a simple TSV list file for bookkeeping:
       data/lists/mid_mini_pairs.tsv
"""
from __future__ import annotations
import argparse, shutil
from pathlib import Path

IMG_EXTS = {".jpg",".JPG"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default="/content/drive/MyDrive/bamr_project")
    ap.add_argument("--convert_raw", action="store_true", default=False,
                    help="If true, convert any .CR2 using rawpy to JPG (requires rawpy).")
    ap.add_argument("--copy", action="store_true", default=True, help="Copy files (default) instead of symlink.")
    args = ap.parse_args()

    PR = Path(args.project_root)
    raw = PR / "data" / "raw" / "mid" / "mini"
    out_img = PR / "data" / "prepared" / "mid_mini" / "images"
    out_img.mkdir(parents=True, exist_ok=True)
    out_list = PR / "data" / "lists" / "mid_mini_pairs.tsv"
    out_list.parent.mkdir(parents=True, exist_ok=True)

    # Optional conversion
    if args.convert_raw:
        try:
            import rawpy, imageio
        except Exception:
            print("[WARN] rawpy not installed; skipping CR2 conversion.")
        else:
            for d in raw.rglob("*"):
                if d.suffix.lower() == ".cr2":
                    jpg = d.with_suffix(".jpg")
                    if jpg.exists(): continue
                    with rawpy.imread(str(d)) as r:
                        rgb = r.postprocess(output_bps=8, no_auto_bright=True)
                        imageio.imwrite(str(jpg), rgb)
                    print("Converted:", d, "->", jpg)

    # Mirror the tree for prepared/images
    rows = []
    for domain in ("Indoor","Outdoor"):
        for pair_dir in sorted((raw/domain).glob("pair*")):
            vA = pair_dir / "viewA"
            vB = pair_dir / "viewB"
            if not vA.exists() or not vB.exists():
                print("[WARN] Skipping (no views):", pair_dir)
                continue
            for v in (vA, vB):
                dst = out_img / domain / pair_dir.name / v.name
                dst.mkdir(parents=True, exist_ok=True)
                for p in sorted(v.glob("*.jpg")) + sorted(v.glob("*.JPG")):
                    q = dst / p.name
                    if args.copy:
                        if not q.exists(): shutil.copy2(p, q)
                    else:
                        try:
                            if not q.exists(): q.symlink_to(p)
                        except Exception:
                            shutil.copy2(p, q)
            # record pairs by common stems
            A = {p.stem: p.name for p in (out_img/domain/pair_dir.name/"viewA").glob("*.jp*g")}
            B = {p.stem: p.name for p in (out_img/domain/pair_dir.name/"viewB").glob("*.jp*g")}
            common = sorted(set(A).intersection(B))
            for s in common:
                rows.append(f"{domain}/{pair_dir.name}\t{A[s]}\t{B[s]}")

    with open(out_list, "w") as f:
        f.write("\n".join(rows))
    print(f"Wrote {len(rows)} pairs -> {out_list}")

if __name__ == "__main__":
    main()
