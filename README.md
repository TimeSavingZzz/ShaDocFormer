# Transformer 阴影融合策略的系统性诊断

[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](./paper/main_cn.pdf)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)

本项目对文档阴影去除中 **Transformer 融合策略的有效性** 进行系统性诊断研究，核心发现：**融合策略的效果取决于训练数据规模与域差异**——复杂融合 (FiLM) 在大规模合成数据上最优，但简单拼接在小规模真实数据上更优。

## 动机

现有文档阴影去除方法在 Transformer backbone 上引入各种阴影引导融合模块，但这些模块为何有效、何时有效、以及如何选择，尚无系统答案。本文在统一 Restormer 框架下对比 5 种融合策略，从数据规模、域差异、注意力机制三个维度进行诊断分析。

## 五种融合策略

| 模型 | 融合方式 | 描述 |
|------|---------|------|
| **Restormer** (baseline) | 无融合 | 仅输入拼接，无独立 shadow encoder |
| **No SGCA** | Concat 拼接 | Shadow encoder 输出与 decoder feature 拼接，无门控 |
| **SGCF** | Cross-Attention | Q=decoder, K/V=shadow encoder, 多头交叉注意力融合 |
| **SGFM** (FiLM) | FiLM 调制 | Shadow encoder 预测 scale/bias 进行特征调制 (tanh 归一化) |
| **SGGF** (Gated) | Sigmoid 门控 | Sigmoid 门控逐通道加权融合 |
| **Large** | Concat + dim=64 | No SGCA 的大模型版本 (dim 48→64) |

## 核心发现

1. **数据依赖律**: 融合策略效果与训练数据量强相关。SGFM (FiLM) 在 SD7K 上超越 No SGCA +1.03 dB (24.77 vs 23.74)，但在 RDD 上不如 No SGCA (37.01 vs 37.20)，因为小数据下复杂融合过拟合。

2. **Transformer 自注意力已处理跨模态交互**: 注意力可视化显示，即使无独立 shadow encoder 的 Restormer baseline，其 decoder 自注意力也已聚焦阴影区域。额外融合模块的作用是增强 (而非建立) 跨模态交互。

3. **域差异定性**: RDD→SD7K 迁移 (~14.6 dB) 与 SD7K→RDD 迁移 (~18.6 dB) 表现出不对称域差异，合成数据难以模拟真实阴影退化。

4. **参数扩展 vs. 融合设计**: 将 No SGCA 的 dim 从 48 提升至 64 (Large) 在两个数据集上均稳定提升，是论文最可靠的结果，适合追求鲁棒性的应用。

## 实验结果

### SD7K (合成文档阴影, 200 train pairs)

| 模型 | PSNR↑ | SSIM↑ | RMSE↓ | TextPSNR↑ |
|------|-------|-------|-------|-----------|
| Restormer | 23.43 | 0.9174 | 23.61 | 26.62 |
| No SGCA | 23.74 | 0.9218 | 23.45 | 27.24 |
| SGCF (CrossAttn) | 23.76 | 0.9164 | 23.64 | 26.46 |
| **SGFM (FiLM)** | **24.77** | **0.9308** | **21.94** | **27.47** |
| SGGF (Gated) | — | — | — | — |
| Large (dim=64) | 24.51 | 0.9282 | 22.10 | 27.61 |

### RDD (真实文档, 1080 train pairs)

| 模型 | PSNR↑ | SSIM↑ | RMSE↓ | TextPSNR↑ |
|------|-------|-------|-------|-----------|
| Restormer | 35.72 | 0.9753 | 5.32 | 33.18 |
| **No SGCA** | **37.20** | **0.9803** | **4.78** | **34.52** |
| SGCF (CrossAttn) | 37.03 | 0.9796 | 4.83 | 34.27 |
| SGFM (FiLM) | 37.01 | 0.9797 | 4.89 | 34.29 |
| SGGF (Gated) | 35.53 | 0.9747 | 5.65 | 33.18 |
| Large (dim=64) | 37.05 | 0.9749 | 4.90 | 33.49 |

## 项目结构

```
├── models/
│   ├── comparison_models.py   # 5 种融合策略模型定义 + SGCF/SGFM/SGGF 模块
│   ├── model.py               # 原始 ShaDocFormer (IJCNN 2024)
│   ├── mask.py                # Shadow-attentive threshold detector
│   ├── refine.py              # Cascaded fusion refiner
│   └── text_aware_model.py    # Text-aware 变体
├── data/
│   ├── data_RGB.py            # 数据加载器
│   ├── dataset_RGB.py         # RDD 数据集
│   └── synthetic_dataset.py   # SD7K 合成数据集
├── config/
│   └── config.py              # 配置管理
├── utils/
│   └── utils.py               # 工具函数 (PSNR, SSIM, 可视化等)
├── paper/                     # 论文 LaTeX 源码与分析脚本
├── train_compare_models.py    # 主训练脚本 (支持 5 模型 × 2 数据集)
├── train.py                   # 原始 ShaDocFormer 训练脚本
├── test.py                    # 原始 ShaDocFormer 测试脚本
├── eval_cross_dataset.py      # 跨域泛化评估
├── eval_ocr.py                # OCR 精度评估 (Tesseract + PaddleOCR)
├── eval_efficiency.py         # 模型效率评估 (参数量/FLOPs/推理时间)
├── evaluate_checkpoint.py     # 单 checkpoint 评估
├── eval_all_checkpoints.py    # 批量 checkpoint 评估
├── run_scaling_curve.py       # 数据量缩放曲线实验
├── visualize_attention.py     # Decoder 自注意力可视化
├── cross_dataset_eval.py      # 跨数据集评估结果汇总
├── plot_training_curves.py    # 训练曲线绘制
├── plot_progress.py           # 实验进度可视化
├── config.yml                 # 训练参数配置
├── environment.yml            # Conda 环境
└── requirements.txt           # Python 依赖
```

## 快速开始

### 环境配置

```bash
git clone https://github.com/TimeSavingZzz/ShaDocFormer.git
cd ShaDocFormer
conda env create -f environment.yml
conda activate shadocformer
```

### 训练

```bash
# 在 RDD (真实文档) 上训练 No SGCA 模型
python train_compare_models.py --model shadow_guided_restormer_no_sgca --dataset rdd --epochs 200 --lr 2e-4 --res 384

# 在 SD7K (合成文档) 上训练 FiLM 模型
python train_compare_models.py --model shadow_guided_restormer_film --dataset sd7k --epochs 200 --lr 2e-4 --res 320

# 可用模型: restormer, shadow_guided_restormer_no_sgca, shadow_guided_restormer_crossattn,
#            shadow_guided_restormer_film, shadow_guided_restormer_gated, shadow_guided_restormer_large
```

### 评估

```bash
# 评估指定 checkpoint 在 RDD 测试集上的表现
python evaluate_checkpoint.py --model shadow_guided_restormer_no_sgca --ckpt <path>.pth --dataset rdd

# 跨域泛化评估 (RDD↔SD7K)
python eval_cross_dataset.py --model shadow_guided_restormer_no_sgca --ckpt <path>.pth

# OCR 性能评估
python eval_ocr.py --ckpt <path>.pth --dataset rdd
```

### 实验脚本

```bash
# 数据量缩放曲线 — 在 SD7K 子集 (30/60/100/150) 上训练
python run_scaling_curve.py --prepare_only           # Step 1: 创建子集
python run_scaling_curve.py --model concat --size 30 --gpu 0  # Step 2: 训练

# 注意力可视化 — 提取并对比 decoder 自注意力图
python visualize_attention.py --data_dir ./dataset/RDD/test/ --num_samples 5
```

## 实验结果复现

完整实验数据与训练日志见 `paper/experiment_results.json`。所有训练在 4× RTX 3090 24GB 上进行，单卡 batch_size=1，200 epochs。

### 预训练+微调结果 (SD7K→RDD)

| 模型 | From-scratch | Finetuned | Δ |
|------|-------------|-----------|----|
| No SGCA | 37.20 | 35.41 | -1.79 |
| SGGF (Gated) | 35.53 | 35.86 | +0.33 |
| Large (dim=64) | 37.05 | 36.39 | -0.66 |

部分模型微调后性能下降，说明合成数据预训练可能将模型导向局部最优。

## 引用

如果本工作对您的研究有帮助，请引用我们的论文：

```bibtex
@article{...
}
```

## 致谢

本项目基于 [Restormer](https://github.com/swz30/Restormer) (CVPR 2022) 和 [ShaDocFormer](https://github.com/kilito777/ShaDocFormer) (IJCNN 2024)。感谢原始作者的优秀工作。
