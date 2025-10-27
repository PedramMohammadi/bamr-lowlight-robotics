# Low‑Light Enhancement for Robotics Applications

Brightness-Aware Mini Restorer (BAMR) is a lightweight image enhancer trained to improve robotics‑relevant downstream metrics (feature matching & object detection) in low light conditions. Unlike classic low‑light methods that optimize PSNR/SSIM only, **BAMR is task‑aware**. after a paired pretraining stage, it is tuned to boost LoFTR matching and YOLO detection.

## 🎯 Goals

Robots often fail in dim environments because visually “prettier” images don’t always help their perception stacks.  
BAMR focuses on **downstream metrics that matter**:

- **Geometric matching** — more keypoint correspondences & inliers (LoFTR).  
- **Detection** — higher mAP on ExDark under low light (YOLOv8).  

The model runs efficiently on embedded GPUs (T4/L4), supports AMP (fp16), and is Colab-friendly.

## What's in this repository
- **Stage‑A**: paired enhancement (LOL‑Blur / LOL‑v2) with L1 + edge + perceptual.
- **Stage‑B**: task‑aware tuning to **maximize matching** (LoFTR) and preserve edges/structure.
- **End‑to‑end eval**: ExDark detector mAP and MID (Indoor/Outdoor) LoFTR matching, with Colab‑ready scripts.

## Headline Results (What I achieved)

### LoFTR on MID-mini (10 pairs × 3 frame-pairs)

| Condition | mean_matches | Δ vs orig | mean_inliers | Δ vs orig |
|:-----------|-------------:|-----------:|-------------:|-----------:|
| Original   | 1132.17 | — | 0.87 | — |
| Stage-A    | 1181.23 | **+4.3%** | 1.00 | **+15%** |
| **Stage-B** | **1277.93** | **+12.9%** | **1.57** | **+81%** |

*Outdoor inliers:* 0.07 → **0.93 (~14×)** with Stage-B.

### YOLOv8n on ExDark (validation)

| Condition | Description | mAP@50 | mAP@50-95 |
|:-----------|:-------------|:------:|:----------:|
| **Enhanced (before FT)** | detector evaluated on Stage-A outputs (no re-training) | 0.582 | 0.356 |
| **Enhanced (after FT)** | detector fine-tuned a few epochs on enhanced data | 0.626 | 0.376 |
| **Stage-B + YOLO adapt** | detector re-trained on Stage-B data | **0.644** | **0.408** |
| **Original (FT)** | fine-tuned on native ExDark | 0.675 | 0.416 |

> **FT = Fine-Tuning** — a short training run on the same (or enhanced) dataset to adapt the YOLO detector to new lighting distributions.

**Takeaway:** Stage-B consistently increases LoFTR robustness and nearly closes the detector gap with the original data after short adaptation.

## ⚙️ Quickstart

### Option A — Colab (recommended)

Open [`notebooks/00_quickstart_colab.py`](notebooks/00_quickstart_colab.py) and run cells top-to-bottom.

It will:
1. Mount Drive  
2. Prepare datasets (YOLO & paired lists)  
3. Train Stage-A / Stage-B  
4. Evaluate YOLO (original vs enhanced)  
5. Run LoFTR before/after comparisons.

### Option B — Local CLI

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Prepare datasets
python scripts/10_prepare_exdark.py --project_root data/raw/exdark
python scripts/11_prepare_lol_blur.py
python scripts/12_prepare_lolv2.py
python scripts/13_prepare_mid_mini.py

# Stage-A (paired pretraining)
python scripts/20_train_stage_a.py --config configs/bamr_stage_a.yaml

# Stage-B (task-aware tuning)
python scripts/30_train_stage_b.py --config configs/bamr_stage_b.yaml

# Detector eval
python detector/yolo_val.py --data configs/exdark_yolo.yaml --name exdark_baseline
python detector/yolo_val.py --data configs/exdark_yolo_enh.yaml --name exdark_enh

# LoFTR eval on MID mini
python matching/loftr_eval_mid.py --project_root . --out results/tables/mid_loftr_before_after.csv
```
## Requirements
- Python ≥ 3.10
- CUDA-capable GPU (T4 or L4)
- See requirements.txt for libraries (Torch, Ultralytics, Kornia, OpenCV, NumPy, Pandas, PyYAML, TQDM)

## 📁 Results Overview
- results_enh_val_before_ft.csv: YOLO val on enhanced (Stage-A) images before fine-tuning
- results_enh_val_after_ft.csv: YOLO val after short fine-tuning (FT) on enhanced images
- results_enh_val_after_taskaware.csv: YOLO val on Stage-B enhanced images (no FT or minimal)
- results_enh_val_after_taskaware_YOLO_adapt.csv: YOLO val after re-training detector on Stage-B data
- mid_loftr_before_after.csv: LoFTR matches & inliers (orig vs Stage-A vs Stage-B)

More details and tables are in results/README.md.

## 🔍 Why This Matters
Traditional enhancers optimize pixel-fidelity metrics (PSNR, SSIM), which don’t always improve perception. **BAMR** directly targets downstream robustness:
- Better geometric alignment between views → stronger visual odometry.
- Improved low-light detection → fewer missed obstacles or objects.
- Lightweight model suitable for embedded deployment.

## 🧠 Highlights & Lessons
- Stage-B loss mix (λ_loftr=1.0, λ_edge=0.6, λ_rc=0.2) delivered the best geometric improvements.
- Mean inliers ↑ 81% overall (MID mini).
- Detector adaptation on enhanced data recovers nearly all mAP lost to distribution shift.
- Simple AMP + stride padding allows stable training on Colab GPUs.

## 💡 Future Work
- Add temporal consistency for video streams.
- Explore mixed fine-tuning (50 % original + 50 % Stage-B images) for detector robustness
- Integrate pose metrics (ATE/RPE) via VO/SLAM downstream tasks.
- Deploy TinyBAMR as an ONNX/TensorRT module on embedded hardware.

## Note: 
Found a bug or have an improvement? Contributions are welcome! 🙌
1. **Open an issue** with a minimal repro (dataset slice, command line, logs).
2. **Submit a Merge Request / Pull Request** referencing the issue.
3. Please follow this checklist:
   - Clear description of the bug/fix
   - Steps to reproduce (commands, args, env details)
   - Affected files

I will review merge requests and leave feedback or merge when ready. Thanks for helping improve the project! 🚀

## Repository layout
```text
bamr-lowlight-robotics/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ .gitignore
├─ bamr/                              # lightweight Python package
│  ├─ __init__.py
│  ├─ models.py                       # TinyBAMR model (Stage‑A/B)
│  ├─ losses.py                       # L1, edge/TV, perceptual, LoFTR/RC terms
│  ├─ data.py                         # ExDark/LOL‑Blur/LOL‑v2/MID datasets + loaders
│  ├─ train_stage_a.py                # paired pretraining (PSNR‑oriented)
│  ├─ train_stage_b.py                # task‑aware tuning (LoFTR/RC)
│  ├─ eval_utils.py                   # PSNR/SSIM, small viz helpers
│  └─ utils.py                        # IO, seeding, logging, amp helpers
├─ detector/
│  ├─ yolo_utils.py                   # wrapper around Ultralytics API
│  ├─ yolo_val.py                     # validate baseline/enhanced sets (JSON/CSV)
│  └─ yolo_finetune.py                # short FT on original/enhanced/mixed
├─ matching/
│  ├─ loftr_eval_mid.py               # memory‑safe “before vs after” MID evaluation
├─ scripts/                           # CLI entrypoints you can run end‑to‑end
│  ├─ 10_prepare_exdark.py
│  ├─ 11_prepare_lol_blur.py
│  ├─ 12_prepare_lolv2.py
│  ├─ 13_prepare_mid_mini.py
│  ├─ 20_train_stage_a.py
│  ├─ 30_train_stage_b.py
│  ├─ 40_eval_yolo_baseline_and_enh.py
│  ├─ 41_yolo_finetune_mix.py
│  └─ 50_loftr_mid_before_after.py
├─ configs/
│  ├─ bamr_stage_a.yaml               # hyperparams for Stage‑A
│  ├─ bamr_stage_b.yaml               # hyperparams for Stage‑B (loss weights, γ)
│  ├─ exdark_yolo.yaml                # original val/test paths
│  ├─ exdark_yolo_enh.yaml            # enhanced val/test paths
│  └─ paths_example.yaml              # example local/Drive paths for data
├─ notebooks/                         # Colab‑first UX
│  ├─ 00_quickstart_colab.py
│  ├─ 01_prepare_datasets.py
│  ├─ 02_train_stage_a.py
│  ├─ 03_train_stage_b.py
│  ├─ 04_yolo_eval_and_ft.py
│  └─ 05_loftr_mid_eval.py
├─ results/
│  ├─ README.md                       # comprehensive results
│  ├─ tables/
│  │  ├─ results_enh_val_before_ft.csv
│  │  ├─ results_enh_val_after_ft.csv
│  │  ├─ results_enh_val_after_taskaware.csv
│  │  ├─ results_enh_val_after_taskaware_YOLO_adapt.csv
│  │  └─ mid_loftr_before_after.csv
│  ├─ figs/
│  │  ├─ EnhancedSamples.png
└─ docs/
   ├─ datasets.md                     # links & expected folder layout for raw data
   └─ design_notes.md                 # brief design choices and limitations
```
