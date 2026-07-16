# Understanding Feature Fusion Strategies in Transformer-based Image Restoration

[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)

Official code for the paper **"Understanding Feature Fusion Strategies in Transformer-based Image Restoration: Evidence from Document Shadow Removal"** (under review, *Applied Intelligence*).

**TL;DR**: We systematically characterize five representative fusion strategies (Concatenation, Cross-Attention, FiLM, Gated, Capacity Scaling) under a controlled Restormer framework across synthetic and real-world document shadow removal datasets. Our key finding: **Transformer self-attention on skip connections already provides implicit cross-modal alignment, making simple concatenation sufficient for real-world data, while complex modulation (FiLM) excels on synthetic data but carries a mechanism-level instability risk on real data.**

---

## Four Key Findings

| Finding | Core Insight |
|---------|-------------|
| **F1: Domain Flipping** | Optimal fusion reverses across domains: FiLM wins on synthetic (24.77 dB), Concat wins on real (37.20 dB). No single best strategy exists. |
| **F2: Dual Constraint Law** | Data domain determines strategy *ranking*; data volume determines gap *magnitude*. FiLM outperforms Concat at all data sizes on synthetic data (no crossover). |
| **F3: Self-Attention as Built-in Fusion** | Decoder self-attention on skip connections already focuses on shadow regions *without* any fusion module. Attention visualization provides direct evidence. |
| **F4: Robustness Hierarchy** | Concat generalizes best (zero-shot SD7K→RDD: 18.59 dB). FiLM collapses to NaN on real data (tanh saturation) — a previously unreported risk. |

---

## Five Fusion Strategies

| Strategy | Type | Params | Mechanism |
|----------|------|--------|-----------|
| **Concat** | Concatenation | +0.30M | Conv([decoder_feat; projected_encoder_feat]) |
| **Cross-Attn** | Attention | +0.44M | Multi-head cross-attention: Q=decoder, KV=shadow |
| **FiLM** | Modulation | +0.34M | Channel-wise F⊙(1+tanh(γ)) + β |
| **Gated** | Gating | +0.29M | Sigmoid-gated convex combination F⊙g + S⊙(1-g) |
| **Large** | Capacity | +20.58M | Concat with backbone dim expanded 48→64 |

All strategies share the same Restormer U-Net backbone, training protocol, and loss function, enabling controlled comparison.

---

## Main Results

### SD7K (Synthetic, 200 training pairs)

| Model | PSNR↑ | SSIM↑ | TextPSNR↑ | LPIPS↓ | Params | Time |
|-------|-------|-------|-----------|--------|--------|------|
| Restormer (no fusion) | 23.43 | 0.9174 | 26.62 | 0.0554 | 26.13M | 181.6ms |
| Concat | 23.74 | 0.9218 | 27.24 | **0.0461** | 26.43M | 181.9ms |
| Cross-Attn | 23.76 | 0.9164 | 26.46 | 0.0480 | 26.57M | 135.0ms |
| **FiLM** | **24.77** | **0.9308** | 27.47 | 0.0557 | 26.47M | 182.5ms |
| Gated | 24.31 | 0.9243 | **27.75** | 0.0496 | 26.42M | 182.6ms |
| Large (dim=64) | 24.51 | 0.9282 | 27.61 | 0.0496 | 46.71M | 346.0ms |

### RDD (Real-world, 1080 training pairs)

| Model | PSNR↑ | SSIM↑ | TextPSNR↑ | LPIPS↓ | Params | Time |
|-------|-------|-------|-----------|--------|--------|------|
| Restormer (no fusion) | 35.72 | 0.9753 | 33.18 | — | 26.13M | 181.6ms |
| **Concat** | **37.20** | **0.9803** | 34.52 | — | 26.43M | 181.9ms |
| Cross-Attn† | 36.99 | 0.9701 | 32.92 | — | 26.57M | 135.0ms |
| FiLM* | 37.01 | 0.9797 | 34.29 | — | 26.47M | 182.5ms |
| Gated | 35.53 | 0.9747 | 33.18 | — | 26.42M | 182.6ms |
| Large (dim=64) | 37.05 | 0.9749 | 33.49 | — | 46.71M | 346.0ms |

† 192×192 due to OOM. \* FiLM v2 with gradient clipping; v1 diverged to NaN at E125/E130.

### Scaling Curve (SD7K subsets, 100 epochs)

| Data Pairs | Concat | FiLM | Gated | FiLM–Concat Gap |
|-----------|--------|------|-------|-----------------|
| 30 | 22.98 | 23.35 | 23.31 | +0.37 |
| 60 | 23.47 | 23.77 | 23.78 | +0.30 |
| 100 | 23.56 | 23.93 | 23.31 | +0.37 |
| 150 | 23.89 | 24.09 | 23.81 | +0.20 |
| 200* | 23.74 | 24.77 | 24.31 | +1.03 |

\*200 pairs at 200 epochs.

---

## Project Structure

```
├── models/
│   ├── comparison_models.py    # 6 model variants (Restormer baseline + 5 fusion)
│   ├── model.py                # Original ShaDocFormer (IJCNN 2024)
│   ├── mask.py                 # Shadow-attentive threshold detector
│   ├── refine.py               # Cascaded fusion refiner
│   └── text_aware_model.py     # Text-aware variant
├── data/
│   ├── data_RGB.py             # Data loader
│   ├── dataset_RGB.py          # RDD dataset
│   └── synthetic_dataset.py    # SD7K synthetic dataset
├── config/
│   └── config.py               # Configuration management
├── utils/
│   └── utils.py                # PSNR, SSIM, visualization utilities
├── paper/
│   └── experiment_results.json # Complete numerical results
├── train_compare_models.py     # Main training script (supports 6 models × 2 datasets)
├── train.py                    # Original ShaDocFormer training script
├── test.py                     # Original ShaDocFormer test script
├── run_scaling_curve.py        # Data scaling curve experiment
├── run_scaling_chain.py        # Sequential scaling job launcher (GPU 1)
├── visualize_attention.py      # Decoder self-attention visualization
├── eval_cross_dataset.py       # Cross-domain generalization evaluation
├── eval_ocr.py                 # OCR accuracy evaluation
├── eval_efficiency.py          # Model efficiency evaluation
├── evaluate_checkpoint.py      # Single checkpoint evaluation
├── eval_all_checkpoints.py     # Batch checkpoint evaluation
├── cross_dataset_eval.py       # Cross-dataset result aggregation
├── plot_training_curves.py     # Training curve plotting
├── plot_progress.py            # Experiment progress visualization
├── config.yml                  # Training parameter configuration
├── environment.yml             # Conda environment
└── requirements.txt            # Python dependencies
```

---

## Quick Start

### Environment

```bash
git clone https://github.com/TimeSavingZzz/ShaDocFormer.git
cd ShaDocFormer
conda env create -f environment.yml
conda activate shadocformer
```

### Training

```bash
# Train Concat model on RDD (real documents)
python train_compare_models.py \
    --model shadow_guided_restormer_no_sgca \
    --dataset rdd --epochs 200 --lr 2e-4 --res 320

# Train FiLM model on SD7K (synthetic documents)
python train_compare_models.py \
    --model shadow_guided_restormer_film \
    --dataset sd7k --epochs 200 --lr 2e-4 --res 320

# Available models: restormer, shadow_guided_restormer_no_sgca,
#   shadow_guided_restormer_crossattn, shadow_guided_restormer_film,
#   shadow_guided_restormer_gated, shadow_guided_restormer_large
```

### Evaluation

```bash
# Single checkpoint evaluation
python evaluate_checkpoint.py --model shadow_guided_restormer_no_sgca \
    --ckpt <path>.pth --dataset rdd

# Cross-domain generalization
python eval_cross_dataset.py --model shadow_guided_restormer_no_sgca \
    --ckpt <path>.pth

# OCR performance
python eval_ocr.py --ckpt <path>.pth --dataset rdd
```

### Analysis Scripts

```bash
# Data scaling curve — train on SD7K subsets (30/60/100/150 pairs)
python run_scaling_curve.py --model concat --size 30 --gpu 0

# Attention visualization — extract and compare decoder self-attention maps
python visualize_attention.py --data_dir ./dataset/RDD/test/ --num_samples 5
```

---

## Reproducibility

All experiments were conducted on 4× NVIDIA RTX 3090 (24GB) with the following unified protocol:
- **Optimizer**: AdamW (β₁=0.9, β₂=0.999), weight decay 10⁻⁴
- **Learning rate**: 2×10⁻⁴, halved at epochs 100 and 150
- **Batch size**: 1 per GPU (gradient accumulation ×2)
- **Epochs**: 200 (100 for scaling curve subsets)
- **Loss**: L₁ + SSIM + FFT frequency-domain loss

Complete numerical results and training logs are available in `paper/experiment_results.json`.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{...,
  title     = {Understanding Feature Fusion Strategies in Transformer-based
               Image Restoration: Evidence from Document Shadow Removal},
  author    = {...},
  journal   = {Applied Intelligence},
  year      = {2026},
  note      = {Under review}
}
```

---

## Acknowledgments

This project builds upon [Restormer](https://github.com/swz30/Restormer) (CVPR 2022) and [ShaDocFormer](https://github.com/kilito777/ShaDocFormer) (IJCNN 2024). We thank the original authors for their excellent work.
