"""FiLM collapse deep analysis — generates data for paper Figures 4-6.

Analysis modules:
  1. Gradient statistics: per-layer gradient norms during forward/backward passes
  2. Activation distributions: tanh input/output histograms on SD7K vs RDD
  3. Modulation saturation: fraction of channels in tanh saturation regime
  4. Feature visualization: PCA/t-SNE of decoder features before/after FiLM
  5. Fisher Information: diagonal approximation to detect early-warning signals

Usage:
  python analyze_film_collapse.py --model_path <checkpoint.pth> \
      --sd7k_root ./dataset/SD7K/test/ \
      --rdd_root ./dataset/RDD/test/ \
      --output ./film_analysis/
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from collections import defaultdict

# ── Dataset ──────────────────────────────────────────────

class PairedImageDataset(Dataset):
    def __init__(self, root, size=256):
        self.size = size
        inp_dir = os.path.join(root, 'img') if os.path.isdir(os.path.join(root, 'img')) else root
        gt_dir = os.path.join(root, 'gt') if os.path.isdir(os.path.join(root, 'gt')) else root
        inp_files = sorted(os.listdir(inp_dir))
        self.samples = []
        for f in inp_files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                gt_f = f
                gt_path = os.path.join(gt_dir, gt_f)
                if os.path.exists(gt_path):
                    self.samples.append((os.path.join(inp_dir, f), gt_path))
                else:
                    # try common gt naming patterns
                    for ext in ['.png', '.jpg', '.jpeg']:
                        alt = os.path.join(gt_dir, os.path.splitext(f)[0] + ext)
                        if os.path.exists(alt):
                            self.samples.append((os.path.join(inp_dir, f), alt))
                            break
        print(f"[Dataset] {root}: {len(self.samples)} pairs found")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp_path, gt_path = self.samples[idx]
        inp = Image.open(inp_path).convert('RGB')
        gt = Image.open(gt_path).convert('RGB')
        inp = inp.resize((self.size, self.size), Image.BILINEAR)
        gt = gt.resize((self.size, self.size), Image.BILINEAR)
        inp_t = torch.from_numpy(np.array(inp).transpose(2, 0, 1)).float() / 255.0
        gt_t = torch.from_numpy(np.array(gt).transpose(2, 0, 1)).float() / 255.0
        gray = 0.299 * inp_t[0] + 0.587 * inp_t[1] + 0.114 * inp_t[2]
        return gray.unsqueeze(0), inp_t, gt_t


# ── Hook-based monitoring ────────────────────────────────

class ActivationMonitor:
    """Register forward/backward hooks to capture activation and gradient stats."""

    def __init__(self, model):
        self.model = model
        self.forward_stats = defaultdict(list)
        self.gradient_stats = defaultdict(list)
        self.handles = []

    def _forward_hook(self, name):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                out_d = out.detach()
                self.forward_stats[f"{name}_mean"].append(out_d.mean().item())
                self.forward_stats[f"{name}_std"].append(out_d.std().item())
                self.forward_stats[f"{name}_min"].append(out_d.min().item())
                self.forward_stats[f"{name}_max"].append(out_d.max().item())
                # Fraction of values in saturation (|x| > 2.0 for tanh)
                sat_frac = (out_d.abs() > 2.0).float().mean().item()
                self.forward_stats[f"{name}_sat_frac"].append(sat_frac)
        return hook

    def _backward_hook(self, name):
        def hook(module, grad_in, grad_out):
            for i, g in enumerate(grad_out):
                if isinstance(g, torch.Tensor):
                    g_d = g.detach()
                    gn = g_d.norm().item()
                    self.gradient_stats[f"{name}_grad_norm"].append(gn)
                    self.gradient_stats[f"{name}_grad_max"].append(g_d.abs().max().item())
                    # Detect exploding gradients
                    if gn > 1000 or torch.isnan(g_d).any():
                        self.gradient_stats[f"{name}_grad_exploded"].append(True)
        return hook

    def register(self, pattern="film"):
        """Register hooks on all modules whose name contains `pattern`."""
        for module_name, module in self.model.named_modules():
            if pattern in module_name.lower():
                self.handles.append(
                    module.register_forward_hook(self._forward_hook(module_name))
                )
                self.handles.append(
                    module.register_full_backward_hook(self._backward_hook(module_name))
                )
        print(f"[Monitor] Registered hooks on {len(self.handles)//2} modules matching '{pattern}'")

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def summarize(self):
        """Compute aggregate statistics across all captured forward passes."""
        summary = {'forward': {}, 'gradient': {}}
        for key, values in self.forward_stats.items():
            arr = np.array(values)
            summary['forward'][key] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'p50': float(np.percentile(arr, 50)),
                'p95': float(np.percentile(arr, 95)),
                'p99': float(np.percentile(arr, 99)),
            }
        for key, values in self.gradient_stats.items():
            arr = np.array([v for v in values if not isinstance(v, bool)])
            if len(arr) > 0:
                summary['gradient'][key] = {
                    'mean': float(np.mean(arr)),
                    'std': float(np.std(arr)),
                    'max': float(np.max(arr)),
                    'p99': float(np.percentile(arr, 99)),
                }
        return summary


# ── Fisher Information (diagonal approximation) ──────────

def compute_fisher_diag(model, dataloader, num_samples=50, device='cuda'):
    """Estimate diagonal Fisher information as empirical gradient squared."""
    model.train()
    fisher = defaultdict(float)
    count = 0

    for batch_idx, (gray, inp, gt) in enumerate(dataloader):
        if count >= num_samples:
            break
        gray, inp, gt = gray.to(device), inp.to(device), gt.to(device)
        model.zero_grad()
        output = model(gray, inp)
        loss = F.l1_loss(output, gt)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += (param.grad.detach() ** 2).mean().item()

        count += 1
        if batch_idx % 10 == 0:
            print(f"  Fisher sample {count}/{num_samples}")

    # Normalize
    for name in fisher:
        fisher[name] /= count
    return dict(fisher)


# ── Feature visualization ────────────────────────────────

def extract_features(model, dataloader, layer_names, device='cuda', max_images=100):
    """Extract intermediate features at specified layers."""
    model.eval()
    features = {name: [] for name in layer_names}
    labels = []  # 0=SD7K, 1=RDD (set by caller)

    def make_hook(name):
        def hook(module, inp, out):
            features[name].append(out.detach().cpu().mean(dim=[2, 3]))  # global avg pool
        return hook

    handles = []
    for module_name, module in model.named_modules():
        for target in layer_names:
            if target in module_name:
                handles.append(module.register_forward_hook(make_hook(module_name)))

    count = 0
    with torch.no_grad():
        for gray, inp, gt in dataloader:
            if count >= max_images:
                break
            gray, inp = gray.to(device), inp.to(device)
            model(gray, inp)
            count += 1

    for h in handles:
        h.remove()

    return {k: torch.cat(v, dim=0).numpy() for k, v in features.items()}


# ── Main analysis ────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='FiLM collapse deep analysis')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained FiLM model checkpoint')
    parser.add_argument('--sd7k_root', type=str, default='./dataset/SD7K/test/')
    parser.add_argument('--rdd_root', type=str, default='./dataset/RDD/test/')
    parser.add_argument('--output', type=str, default='./film_analysis/')
    parser.add_argument('--model_type', type=str, default='shadow_guided_restormer_film',
                        help='Model class key for build_model()')
    parser.add_argument('--num_images', type=int, default=100,
                        help='Number of images per dataset for statistics')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Import model builder from the project
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, '/mnt/ShaDocFormer-main')
    from train_compare_models import build_model

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    # Load model
    print(f"[Model] Building {args.model_type}...")
    model = build_model(args.model_type)
    ckpt = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in ckpt:
        ckpt = ckpt['model_state_dict']
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()
    print(f"[Model] Loaded from {args.model_path}")

    # ── 1. Activation distribution analysis ────────────────
    print("\n" + "="*60)
    print("ANALYSIS 1: Activation Distributions (SD7K vs RDD)")
    print("="*60)

    monitor = ActivationMonitor(model)
    monitor.register(pattern="film")
    monitor.register(pattern="tanh")
    monitor.register(pattern="modulation")

    for dataset_name, root in [('SD7K', args.sd7k_root), ('RDD', args.rdd_root)]:
        print(f"\n--- {dataset_name} ---")
        ds = PairedImageDataset(root)
        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

        monitor.forward_stats.clear()
        monitor.gradient_stats.clear()

        with torch.no_grad():
            for i, (gray, inp, gt) in enumerate(dl):
                if i >= args.num_images:
                    break
                gray, inp = gray.to(device), inp.to(device)
                try:
                    model(gray, inp)
                except Exception as e:
                    print(f"  ERROR at image {i}: {e}")
                    break

        summary = monitor.summarize()
        with open(os.path.join(args.output, f'activation_{dataset_name.lower()}.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved activation_{dataset_name.lower()}.json")

    monitor.remove()

    # ── 2. Gradient statistics ────────────────────────────
    print("\n" + "="*60)
    print("ANALYSIS 2: Gradient Norm Statistics")
    print("="*60)

    monitor2 = ActivationMonitor(model)
    monitor2.register(pattern="film")
    monitor2.register(pattern="tanh")
    monitor2.register(pattern="modulation")

    for dataset_name, root in [('SD7K', args.sd7k_root), ('RDD', args.rdd_root)]:
        print(f"\n--- {dataset_name} ---")
        ds = PairedImageDataset(root)
        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

        monitor2.forward_stats.clear()
        monitor2.gradient_stats.clear()

        model.train()
        for i, (gray, inp, gt) in enumerate(dl):
            if i >= min(args.num_images, 30):  # limited for backward passes (memory)
                break
            gray, inp, gt = gray.to(device), inp.to(device), gt.to(device)
            model.zero_grad()
            try:
                output = model(gray, inp)
                loss = F.l1_loss(output, gt)
                loss.backward()

                # Check for NaN
                has_nan = False
                for name, param in model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        has_nan = True
                        print(f"  NaN in {name}.grad at image {i}!")
                        break
                if has_nan:
                    break
            except Exception as e:
                print(f"  ERROR at image {i}: {e}")
                break

        summary = monitor2.summarize()
        with open(os.path.join(args.output, f'gradient_{dataset_name.lower()}.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved gradient_{dataset_name.lower()}.json")

    monitor2.remove()

    # ── 3. Fisher Information ─────────────────────────────
    print("\n" + "="*60)
    print("ANALYSIS 3: Diagonal Fisher Information")
    print("="*60)

    for dataset_name, root in [('SD7K', args.sd7k_root), ('RDD', args.rdd_root)]:
        print(f"\n--- {dataset_name} ---")
        ds = PairedImageDataset(root)
        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

        fisher = compute_fisher_diag(model, dl, num_samples=30, device=device)
        # Filter to modulation-related parameters
        fisher_mod = {k: v for k, v in fisher.items()
                      if any(p in k.lower() for p in ['film', 'tanh', 'modulation', 'gamma', 'beta'])}
        with open(os.path.join(args.output, f'fisher_{dataset_name.lower()}.json'), 'w') as f:
            json.dump(fisher_mod, f, indent=2)
        print(f"  Fisher entries: {len(fisher_mod)} modulation params")

    # ── 4. Feature visualization ──────────────────────────
    print("\n" + "="*60)
    print("ANALYSIS 4: Feature Visualization (PCA-ready)")
    print("="*60)

    # Find relevant layer names
    layer_names = []
    for name, _ in model.named_modules():
        if 'film' in name.lower() or 'decoder_level' in name.lower():
            layer_names.append(name)

    print(f"  Target layers: {len(layer_names)}")

    for dataset_name, root in [('SD7K', args.sd7k_root), ('RDD', args.rdd_root)]:
        print(f"\n--- {dataset_name} ---")
        ds = PairedImageDataset(root)
        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

        feats = extract_features(model, dl, layer_names, device=device, max_images=args.num_images)
        # Save as float32 numpy arrays
        np.savez_compressed(
            os.path.join(args.output, f'features_{dataset_name.lower()}.npz'),
            **feats
        )
        print(f"  Saved features_{dataset_name.lower()}.npz")

    # ── Summary ───────────────────────────────────────────
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to {args.output}/")
    print("Files:")
    for f in sorted(os.listdir(args.output)):
        size_mb = os.path.getsize(os.path.join(args.output, f)) / 1024 / 1024
        print(f"  {f} ({size_mb:.1f} MB)")
    print("\nNext: use plot_film_analysis.py to generate paper figures.")


if __name__ == '__main__':
    main()
