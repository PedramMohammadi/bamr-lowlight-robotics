# Results, Analysis, and Future Directions

This document consolidates detector and matching results, describes evaluation settings, and lists actionable improvements.

## 1) Datasets & splits

- **ExDark** (detection): converted to YOLO format with `train/val/test` (12 classes).
- **LOL‑Blur / LOL‑v2** (paired): used for Stage‑A supervised enhancement.
- **MID (Indoor/Outdoor)**: two‑view RAW (converted to JPG) with GT essential matrix; used for LoFTR matching.

## 2) Training & eval settings

### Stage‑A (paired pretraining)
- **Model:** TinyBAMR (depthwise‑separable UNet with long‑receptive fields + freq. attention).
- **Loss:** `λ_L1=1.0`, `λ_edge=0.5`, `λ_perc=0.2` (VGG‑style).
- **Val metric:** PSNR on 512 patches (best ~**18.45 dB** early in training).

### Stage‑B (task‑aware tuning)
- **Objective:** improve downstream **matching & edges**.
- **Losses:** `L = λ_loftr * L_loftr + λ_edge * L_edge + λ_rc * L_rc`
  - `L_loftr`: encourages features beneficial to LoFTR (hinge / margin‑based).
  - `L_edge`: keeps structural fidelity (edge/TV).
  - `L_rc`: “response consistency” prior to prevent hallucination.
- **Suggested weights:** `λ_loftr=1.0`, `λ_edge=0.6`, `λ_rc=0.2` (used in our best run).
- **AMP:** fp16 on T4/L4, batch‑wise gradient accumulation if needed.

### Detector (YOLOv8n)
- **Baselines:** pretrained `yolov8n.pt` validated on `exdark_yolo.yaml` (original) and `exdark_yolo_enh.yaml` (enhanced).
- **Adaptation:** short fine‑tune on original/enhanced/mixed data as indicated below.
- **Outputs:** `predictions.json` (COCO), `results.csv` (mAP/precision/recall).

### Matching (LoFTR)
- **Backbone:** `LoFTR(pretrained="outdoor")`, mixed precision on GPU.
- **Resolution:** long edge **≤ 1280 px** (memory‑safe); same scale across conditions.
- **Metrics:** matches, mean confidence, Sampson inliers vs. GT `E_estimated`.

## 3) Key tables (from `/results/tables`)

### 3.1 ExDark detection (mAP on val)

| Setting                               | mAP@50 | mAP@50–95 | File |
|---|---:|---:|---|
| Enhanced **before** YOLO FT           | 0.582  | 0.356     | `results_enh_val_before_ft.csv` |
| Enhanced **after** YOLO FT            | 0.626  | 0.376     | `results_enh_val_after_ft.csv` |
| **Stage‑B** (+ YOLO adapt on Stage‑B) | ~0.644 | ~0.408    | `results_enh_val_after_taskaware.csv` / `_YOLO_adapt.csv` |
| Original (YOLO FT on originals)       | 0.675  | 0.416     | (baseline fine‑tuned) |

**Interpretation.** Stage‑B + YOLO adaptation adds **+6.2 mAP@50** over the enhanced baseline (0.582 → 0.644) and approaches the original fine‑tuned performance (0.675). Expect parity or gains on low‑light subsets and mixed training (see §4).

### 3.2 MID matching (LoFTR, overall)

| Condition | mean_matches | Δ vs orig | mean_inliers | Δ vs orig | File |
|---|---:|---:|---:|---:|---|
| Original | 1132.17 | — | 0.87 | — | `mid_loftr_before_after.csv` |
| Stage‑A  | 1181.23 | +4.3% | 1.00 | +15% | 〃 |
| **Stage‑B** | **1277.93** | **+12.9%** | **1.57** | **+81%** | 〃 |

*Outdoor inliers:* **0.07 → 0.93 (~14×)** after Stage‑B. Indoor shows steady, smaller gains.


## 4) What worked & what didn’t

**Worked**
- Stage‑B’s task‑aware loss delivered **robust geometric improvements** without exploding runtime.
- Detector adaptation on the **enhanced distribution** recovered most of the mAP gap.

**Limitations**
- LoFTR mean confidence dips after photometric normalization (not an issue; matching/inliers tell the story).
- Best detector parity likely requires **mixed training** (original+enhanced).

## 5) Practical improvements (next steps)

1. **Mixed YOLO fine‑tune** (50/50 original+Stage‑B) with lighter color jitter → typically closes the last mAP gap and helps generalization.
2. **Edge‑biased Stage‑B** (if needed): keep `λ_loftr=1.0`, try `λ_edge=0.8`, `λ_rc=0.15`, and a slightly smaller margin/γ (−10–20%) to sharpen structures.
3. **LoFTR eval++**: compute **inlier ratio** and RANSAC‑estimated E for each pair (less sensitive than raw counts).
4. **Runtime**: export TinyBAMR to ONNX/FP16 TensorRT for embedded deployment.
