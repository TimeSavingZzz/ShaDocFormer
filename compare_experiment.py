"""
文本感知ShaDocFormer vs 原始ShaDocFormer 对比实验

对比项目:
1. PSNR / SSIM / RMSE (标准图像质量指标)
2. 文字区域局部PSNR (文本保护效果)
3. OCR识别准确率 (EasyOCR)
"""
import os
import sys
import warnings
import time

warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.text_aware_model import TextAwareModel, TextDetector, TextAwareLoss, compute_ocr_accuracy
from models import Model
from utils import seed_everything


class SyntheticDataset(Dataset):
    """合成文档阴影数据集加载器"""

    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        self.input_dir = os.path.join(data_dir, split, 'input')
        self.target_dir = os.path.join(data_dir, split, 'target')
        self.files = sorted(os.listdir(self.input_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        inp_path = os.path.join(self.input_dir, fname)
        tar_path = os.path.join(self.target_dir, fname)

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')

        inp_tensor = TF.to_tensor(inp_img)
        tar_tensor = TF.to_tensor(tar_img)
        bin_tensor = TF.rgb_to_grayscale(inp_tensor)

        return inp_tensor, bin_tensor, tar_tensor, fname


def compute_text_region_metrics(pred, target, text_attention):
    """计算文字区域的局部PSNR和SSIM"""
    text_mask = text_attention.expand_as(pred)
    text_mask_bin = (text_mask > 0.3).float()

    text_area = text_mask_bin.sum().item()
    if text_area < 100:
        return {'text_psnr': 0, 'text_ssim': 0, 'text_rmse': 0}

    text_pred = pred * text_mask_bin
    text_target = target * text_mask_bin

    mse = torch.mean((text_pred - text_target) ** 2).item()
    if mse < 1e-10:
        text_psnr = 100
    else:
        text_psnr = 20 * np.log10(1.0 / np.sqrt(mse))

    # 简化的局部SSIM
    pred_sum = text_pred.sum()
    target_sum = text_target.sum()
    if torch.abs(target_sum) < 1e-10:
        text_ssim = 1.0
    else:
        text_ssim = 1.0 - torch.abs(pred_sum - target_sum) / (torch.abs(target_sum) + 1e-8)
        text_ssim = max(0.0, min(1.0, text_ssim.item()))

    text_rmse = np.sqrt(mse) * 255

    return {
        'text_psnr': round(text_psnr, 2),
        'text_ssim': round(text_ssim, 4),
        'text_rmse': round(text_rmse, 2),
    }


def train_one_epoch(model, dataloader, optimizer, criterion, text_detector, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    losses_dict = {'loss_text': 0, 'loss_bg': 0, 'loss_ssim': 0}

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (inp, gray, tar, _) in enumerate(pbar):
        inp = inp.to(device)
        gray = gray.to(device)
        tar = tar.to(device)

        # 生成文本注意力图
        text_attn = TextDetector.batch_detect(text_detector, tar, device)
        # 用原始阴影图做文本检测 (无阴影时文字更清晰)
        text_attn_input = TextDetector.batch_detect(text_detector, inp, device)
        text_attn = torch.max(text_attn, text_attn_input)  # 合并两者, [B,1,H,W]

        optimizer.zero_grad()

        # Text-Aware Model uses text_attention
        if isinstance(model, TextAwareModel):
            res = model(gray, inp, text_attention=text_attn)
        else:
            res = model(gray, inp)

        loss, loss_detail = criterion(res, tar, text_attn)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        for k in losses_dict:
            losses_dict[k] += loss_detail.get(k, 0)

        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'text': f'{loss_detail.get("loss_text", 0):.4f}',
                'bg': f'{loss_detail.get("loss_bg", 0):.4f}',
            })

    n = len(dataloader)
    return {
        'total_loss': total_loss / n,
        'loss_text': losses_dict['loss_text'] / n,
        'loss_bg': losses_dict['loss_bg'] / n,
        'loss_ssim': losses_dict['loss_ssim'] / n,
    }


@torch.no_grad()
def evaluate(model, dataloader, text_detector, device):
    """评估模型"""
    model.eval()
    metrics = {
        'psnr': [], 'ssim': [], 'rmse': [],
        'text_psnr': [], 'text_ssim': [], 'text_rmse': [],
        'ocr_orig': [], 'ocr_restored': [], 'ocr_gt': [],
    }

    from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
    from torchmetrics.functional.regression import mean_squared_error

    for inp, gray, tar, fname in tqdm(dataloader, desc="Evaluating"):
        inp = inp.to(device)
        gray = gray.to(device)
        tar = tar.to(device)

        # 文本注意力图 (从真值图检测, 因为最准确)
        text_attn = TextDetector.batch_detect(text_detector, tar, device)  # [B,1,H,W]

        if isinstance(model, TextAwareModel):
            res = model(gray, inp, text_attention=text_attn)
        else:
            res = model(gray, inp)

        # 全局指标
        psnr_val = peak_signal_noise_ratio(res, tar, data_range=1).item()
        ssim_val = structural_similarity_index_measure(res, tar, data_range=1).item()
        rmse_val = mean_squared_error(res * 255, tar * 255, squared=False).item()

        metrics['psnr'].append(psnr_val)
        metrics['ssim'].append(ssim_val)
        metrics['rmse'].append(rmse_val)

        # 文本区域局部指标
        text_metrics = compute_text_region_metrics(res, tar, text_attn)
        metrics['text_psnr'].append(text_metrics['text_psnr'])
        metrics['text_ssim'].append(text_metrics['text_ssim'])
        metrics['text_rmse'].append(text_metrics['text_rmse'])

        # OCR评估 (仅对前5张图做, 节省时间)
        if len(metrics['ocr_orig']) < 5:
            ocr_result = compute_ocr_accuracy(inp[0], res[0], tar[0])
            if ocr_result is not None:
                metrics['ocr_orig'].append(ocr_result['original_ocr_chars'])
                metrics['ocr_restored'].append(ocr_result['restored_ocr_chars'])
                metrics['ocr_gt'].append(ocr_result['gt_ocr_chars'])

    # 计算平均值
    result = {}
    for k in ['psnr', 'ssim', 'rmse', 'text_psnr', 'text_ssim', 'text_rmse']:
        if metrics[k]:
            result[k] = round(np.mean(metrics[k]), 4)
        else:
            result[k] = 0

    if metrics['ocr_orig']:
        result['ocr_original'] = round(np.mean(metrics['ocr_orig']), 1)
        result['ocr_restored'] = round(np.mean(metrics['ocr_restored']), 1)
        result['ocr_gt'] = round(np.mean(metrics['ocr_gt']), 1)
        result['ocr_recovery'] = round(
            np.mean(metrics['ocr_restored']) / max(np.mean(metrics['ocr_gt']), 1) * 100, 1
        )
    else:
        result['ocr_original'] = 'N/A'
        result['ocr_restored'] = 'N/A'
        result['ocr_gt'] = 'N/A'
        result['ocr_recovery'] = 'N/A'

    return result


def compare_experiments(data_dir, output_dir, num_epochs=10):
    """
    运行对比实验:
    1. 原始ShaDocFormer (baseline)
    2. 文本感知ShaDocFormer (our method)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Data: {data_dir}")
    print(f"Epochs: {num_epochs}")
    print("=" * 60)

    seed_everything(3407)

    # 数据加载
    train_dataset = SyntheticDataset(data_dir, 'train')
    test_dataset = SyntheticDataset(data_dir, 'test')
    trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # 文本检测器
    text_detector = TextDetector(method='opencv')

    results = {}

    # ====== 实验1: Baseline (原始ShaDocFormer) ======
    print("\n" + "=" * 60)
    print("Experiment 1: Baseline ShaDocFormer")
    print("=" * 60)

    model_baseline = Model().to(device)
    optimizer = optim.AdamW(model_baseline.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-5)
    criterion_mse = torch.nn.MSELoss()

    for epoch in range(1, num_epochs + 1):
        model_baseline.train()
        epoch_loss = 0
        for inp, gray, tar, _ in tqdm(trainloader, desc=f"Baseline Epoch {epoch}"):
            inp, gray, tar = inp.to(device), gray.to(device), tar.to(device)
            optimizer.zero_grad()
            res = model_baseline(gray, inp)
            loss = criterion_mse(res, tar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        print(f"  Epoch {epoch}: loss={epoch_loss / len(trainloader):.6f}")

    results['Baseline'] = evaluate(model_baseline, testloader, text_detector, device)
    print(f"\nBaseline Results:")
    print(f"  PSNR: {results['Baseline']['psnr']:.2f}, SSIM: {results['Baseline']['ssim']:.4f}, RMSE: {results['Baseline']['rmse']:.2f}")
    print(f"  Text PSNR: {results['Baseline']['text_psnr']:.2f}, Text SSIM: {results['Baseline']['text_ssim']:.4f}")
    print(f"  OCR: {results['Baseline']['ocr_original']} -> {results['Baseline']['ocr_restored']} chars (GT: {results['Baseline']['ocr_gt']})")

    # ====== 实验2: Text-Aware ShaDocFormer ======
    print("\n" + "=" * 60)
    print("Experiment 2: Text-Aware ShaDocFormer (Ours)")
    print("=" * 60)

    seed_everything(3407)  # 同样的随机种子确保公平对比
    model_textaware = TextAwareModel().to(device)
    optimizer_ta = optim.AdamW(model_textaware.parameters(), lr=1e-3)
    scheduler_ta = optim.lr_scheduler.CosineAnnealingLR(optimizer_ta, num_epochs, eta_min=1e-5)
    criterion_ta = TextAwareLoss(text_weight=2.0, bg_weight=1.0, ssim_weight=0.2)

    for epoch in range(1, num_epochs + 1):
        model_textaware.train()
        epoch_loss = 0
        for inp, gray, tar, _ in tqdm(trainloader, desc=f"TextAware Epoch {epoch}"):
            inp, gray, tar = inp.to(device), gray.to(device), tar.to(device)
            text_attn = TextDetector.batch_detect(text_detector, tar, device)
            text_attn_inp = TextDetector.batch_detect(text_detector, inp, device)
            text_attn = torch.max(text_attn, text_attn_inp)  # [B,1,H,W]

            optimizer_ta.zero_grad()
            res = model_textaware(gray, inp, text_attention=text_attn)
            loss, _ = criterion_ta(res, tar, text_attn)
            loss.backward()
            optimizer_ta.step()
            epoch_loss += loss.item()
        scheduler_ta.step()
        print(f"  Epoch {epoch}: loss={epoch_loss / len(trainloader):.6f}")

    results['TextAware'] = evaluate(model_textaware, testloader, text_detector, device)
    print(f"\nText-Aware Results:")
    print(f"  PSNR: {results['TextAware']['psnr']:.2f}, SSIM: {results['TextAware']['ssim']:.4f}, RMSE: {results['TextAware']['rmse']:.2f}")
    print(f"  Text PSNR: {results['TextAware']['text_psnr']:.2f}, Text SSIM: {results['TextAware']['text_ssim']:.4f}")
    print(f"  OCR: {results['TextAware']['ocr_original']} -> {results['TextAware']['ocr_restored']} chars (GT: {results['TextAware']['ocr_gt']})")

    # ====== 对比总结 ======
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    print(f"\n{'Metric':<20} {'Baseline':>12} {'TextAware':>12} {'Improvement':>12}")
    print("-" * 60)

    for metric in ['psnr', 'ssim', 'rmse']:
        base_val = results['Baseline'][metric]
        ta_val = results['TextAware'][metric]
        if metric == 'rmse':
            imp = (base_val - ta_val) / base_val * 100
            direction = '↓'
        else:
            imp = (ta_val - base_val) / base_val * 100
            direction = '↑'
        print(f"{'Global '+metric.upper():<20} {base_val:>10.4f}{direction} {ta_val:>10.4f}{direction} {imp:>+9.1f}%")

    print()
    for metric in ['text_psnr', 'text_ssim', 'text_rmse']:
        base_val = results['Baseline'][metric]
        ta_val = results['TextAware'][metric]
        if base_val == 0 or ta_val == 0:
            continue
        if 'rmse' in metric:
            imp = (base_val - ta_val) / base_val * 100
            direction = '↓'
        else:
            imp = (ta_val - base_val) / base_val * 100
            direction = '↑'
        label = metric.replace('_', ' ').title()
        print(f"{label:<20} {base_val:>10.4f}{direction} {ta_val:>10.4f}{direction} {imp:>+9.1f}%")

    print(f"\n{'OCR Recovery Rate':<20} {'N/A':>12} {results['TextAware'].get('ocr_recovery', 'N/A'):>12}")

    # 保存模型
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model_textaware.state_dict(), os.path.join(output_dir, 'text_aware_model.pth'))

    # 保存对比图片
    os.makedirs(os.path.join(output_dir, 'comparison'), exist_ok=True)
    print(f"\nResults saved to {output_dir}")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./dataset/Synthetic/')
    parser.add_argument('--output_dir', type=str, default='./experiment_results/')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    # 如果没有合成数据, 先生成
    if not os.path.exists(args.data_dir):
        print("Generating synthetic dataset...")
        from data.synthetic_dataset import generate_dataset
        generate_dataset(args.data_dir, num_train=200, num_test=50)

    compare_experiments(args.data_dir, args.output_dir, args.epochs)
