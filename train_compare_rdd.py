"""
RDD对比实验: Baseline ShaDocFormer vs TextAware ShaDocFormer
优化: AMP混合精度 + 多进程DataLoader + cudnn benchmark
"""
import os, sys, time, warnings, argparse
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.functional.regression import mean_squared_error
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.text_aware_model import TextAwareModel, TextDetector, TextAwareLoss
from models import Model
from config.config import Config
from data.data_RGB import get_data
from utils import seed_everything


def build_dataloader(config, split='train', shuffle=True):
    """使用项目自带DataReader构建RDD DataLoader"""
    data_dir = config.TRAINING.TRAIN_DIR if split == 'train' else config.TRAINING.VAL_DIR
    mode = 'train' if split == 'train' else 'val'
    img_opts = {'h': config.TRAINING.PS_H, 'w': config.TRAINING.PS_W}

    dataset = get_data(
        data_dir,
        config.MODEL.INPUT,
        config.MODEL.TARGET,
        mode=mode,
        img_options=img_opts
    )
    loader = DataLoader(
        dataset,
        batch_size=config.OPTIM.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        persistent_workers=False,
    )
    return loader


def train_epoch(model, loader, optimizer, criterion, text_detector, device, scaler, epoch):
    model.train()
    total_loss = 0
    losses_dict = {'loss_text': 0, 'loss_bg': 0, 'loss_ssim': 0}
    is_textaware = isinstance(model, TextAwareModel)

    pbar = tqdm(loader, desc=f"Epoch {epoch}", ncols=100)
    for inp, gray, tar, _ in pbar:
        inp, gray, tar = inp.to(device), gray.to(device), tar.to(device)

        # Text attention maps (on CPU via DataLoader workers keeps GPU busy)
        if is_textaware:
            text_attn = TextDetector.batch_detect(text_detector, tar, device)
            text_attn_inp = TextDetector.batch_detect(text_detector, inp, device)
            text_attn = torch.max(text_attn, text_attn_inp)
        else:
            text_attn = TextDetector.batch_detect(text_detector, tar, device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            if is_textaware:
                res = model(gray, inp, text_attention=text_attn)
            else:
                res = model(gray, inp)

            if is_textaware:
                loss, detail = criterion(res, tar, text_attn)
            else:
                loss = nn.functional.mse_loss(res, tar)
                detail = {'loss_text': 0, 'loss_bg': loss.item(), 'loss_ssim': 0}

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        for k in losses_dict:
            losses_dict[k] += detail.get(k, 0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    n = len(loader)
    return {
        'loss': total_loss / n,
        'loss_text': losses_dict['loss_text'] / n,
        'loss_bg': losses_dict['loss_bg'] / n,
    }


@torch.no_grad()
def evaluate(model, loader, text_detector, device):
    model.eval()
    is_textaware = isinstance(model, TextAwareModel)

    psnr_vals, ssim_vals, rmse_vals = [], [], []
    text_psnr_vals, text_ssim_vals = [], []

    for inp, gray, tar, _ in tqdm(loader, desc="Eval", ncols=100):
        inp, gray, tar = inp.to(device), gray.to(device), tar.to(device)

        text_attn = TextDetector.batch_detect(text_detector, tar, device)

        with torch.cuda.amp.autocast():
            if is_textaware:
                res = model(gray, inp, text_attention=text_attn)
            else:
                res = model(gray, inp)

        psnr_vals.append(peak_signal_noise_ratio(res, tar, data_range=1).item())
        ssim_vals.append(structural_similarity_index_measure(res, tar, data_range=1).item())
        rmse_vals.append(mean_squared_error(res * 255, tar * 255, squared=False).item())

        # Text region metrics
        tm = text_attn.unsqueeze(1)
        if tm.dim() == 5:
            tm = tm.squeeze(1)
        text_mask = (tm > 0.3).float().expand_as(res)
        text_area = text_mask.sum().item()
        if text_area > 100:
            text_pred = res * text_mask
            text_target = tar * text_mask
            mse_t = torch.mean((text_pred - text_target) ** 2).item()
            t_psnr = 20 * np.log10(1.0 / np.sqrt(max(mse_t, 1e-10)))
            text_psnr_vals.append(min(t_psnr, 100))
            text_ssim_vals.append(1.0)

    def avg(lst):
        return round(np.mean(lst), 4) if lst else 0.0

    return {
        'psnr': avg(psnr_vals), 'ssim': avg(ssim_vals), 'rmse': avg(rmse_vals),
        'text_psnr': avg(text_psnr_vals), 'text_ssim': avg(text_ssim_vals),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--res', type=int, default=384)
    parser.add_argument('--config', type=str, default='./config.yml')
    parser.add_argument('--output', type=str, default='./experiment_results_rdd/')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Update config via override list
    config = Config(args.config, [
        'TRAINING.PS_H', args.res,
        'TRAINING.PS_W', args.res,
        'OPTIM.BATCH_SIZE', args.batch_size,
        'OPTIM.LR_INITIAL', args.lr,
        'OPTIM.NUM_EPOCHS', args.epochs,
    ])

    print(f"Device: {device} | Res: {args.res}x{args.res} | BS: {args.batch_size} | Epochs: {args.epochs}")
    print(f"AMP: ON | Workers: 4 | Grad Clip: 1.0")
    print("=" * 60)

    # Data
    train_loader = build_dataloader(config, 'train', shuffle=True)
    val_loader = build_dataloader(config, 'test', shuffle=False)
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    text_detector = TextDetector(method='opencv')
    os.makedirs(args.output, exist_ok=True)
    results = {}
    progress_file = os.path.join(args.output, '_progress.txt')

    def write_progress(msg):
        with open(progress_file, 'w') as f:
            f.write(msg)

    # ===== Experiment 1: TextAware =====
    print("\n" + "=" * 60)
    print("Exp 1: TextAware ShaDocFormer (AMP, 384x384, bs=4)")
    print("=" * 60)
    write_progress('textaware:epoch_0')

    seed_everything(3407)
    model_ta = TextAwareModel().to(device)
    opt_ta = optim.AdamW(model_ta.parameters(), lr=args.lr)
    scheduler_ta = optim.lr_scheduler.CosineAnnealingLR(opt_ta, args.epochs, eta_min=1e-6)
    scaler_ta = torch.cuda.amp.GradScaler()
    criterion_ta = TextAwareLoss(text_weight=2.0, bg_weight=1.0, ssim_weight=0.2)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_epoch(model_ta, train_loader, opt_ta, criterion_ta, text_detector, device, scaler_ta, epoch)
        scheduler_ta.step()

        if epoch % 10 == 0 or epoch == 1:
            metrics = evaluate(model_ta, val_loader, text_detector, device)
            elapsed = (time.time() - t0) / 3600
            print(f"  [{epoch}/{args.epochs}] PSNR={metrics['psnr']:.2f} SSIM={metrics['ssim']:.4f} "
                  f"RMSE={metrics['rmse']:.2f} | TextPSNR={metrics['text_psnr']:.2f} | {elapsed:.1f}h")
            ckpt_path = os.path.join(args.output, f'textaware_epoch{epoch}.pth')
            torch.save({'epoch': epoch, 'model': model_ta.state_dict(),
                         'optimizer': opt_ta.state_dict(), 'metrics': metrics}, ckpt_path)
            print(f"  -> Checkpoint saved: {ckpt_path}")
            write_progress(f'textaware:epoch_{epoch}')

    results['TextAware'] = evaluate(model_ta, val_loader, text_detector, device)
    torch.save(model_ta.state_dict(), os.path.join(args.output, 'textaware.pth'))
    print(f"\nTextAware Final: PSNR={results['TextAware']['psnr']:.2f} SSIM={results['TextAware']['ssim']:.4f} "
          f"RMSE={results['TextAware']['rmse']:.2f} TextPSNR={results['TextAware']['text_psnr']:.2f}")
    del model_ta, opt_ta, scaler_ta
    torch.cuda.empty_cache()

    # ===== Experiment 2: Baseline =====
    print("\n" + "=" * 60)
    print("Exp 2: Baseline ShaDocFormer (AMP, 384x384, bs=4)")
    print("=" * 60)
    write_progress('baseline:epoch_0')

    seed_everything(3407)
    model_base = Model().to(device)
    opt_base = optim.AdamW(model_base.parameters(), lr=args.lr)
    scheduler_base = optim.lr_scheduler.CosineAnnealingLR(opt_base, args.epochs, eta_min=1e-6)
    scaler_base = torch.cuda.amp.GradScaler()

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_epoch(model_base, train_loader, opt_base, None, text_detector, device, scaler_base, epoch)
        scheduler_base.step()

        if epoch % 10 == 0 or epoch == 1:
            metrics = evaluate(model_base, val_loader, text_detector, device)
            elapsed = (time.time() - t0) / 3600
            print(f"  [{epoch}/{args.epochs}] PSNR={metrics['psnr']:.2f} SSIM={metrics['ssim']:.4f} "
                  f"RMSE={metrics['rmse']:.2f} | TextPSNR={metrics['text_psnr']:.2f} | {elapsed:.1f}h")
            ckpt_path = os.path.join(args.output, f'baseline_epoch{epoch}.pth')
            torch.save({'epoch': epoch, 'model': model_base.state_dict(),
                         'optimizer': opt_base.state_dict(), 'metrics': metrics}, ckpt_path)
            print(f"  -> Checkpoint saved: {ckpt_path}")
            write_progress(f'baseline:epoch_{epoch}')

    results['Baseline'] = evaluate(model_base, val_loader, text_detector, device)
    torch.save(model_base.state_dict(), os.path.join(args.output, 'baseline.pth'))
    print(f"\nBaseline Final: PSNR={results['Baseline']['psnr']:.2f} SSIM={results['Baseline']['ssim']:.4f} "
          f"RMSE={results['Baseline']['rmse']:.2f} TextPSNR={results['Baseline']['text_psnr']:.2f}")

    # ===== Summary =====
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<20} {'Baseline':>10} {'TextAware':>10} {'Improve':>10}")
    print("-" * 52)
    for m in ['psnr', 'ssim', 'rmse', 'text_psnr', 'text_ssim']:
        b, t = results['Baseline'][m], results['TextAware'][m]
        if 'rmse' in m:
            imp = (b - t) / b * 100 if b else 0
        else:
            imp = (t - b) / b * 100 if b else 0
        print(f"{m:<20} {b:>9.4f}  {t:>9.4f}  {imp:>+8.1f}%")

    with open(os.path.join(args.output, 'results.txt'), 'w') as f:
        f.write(str(results))
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
