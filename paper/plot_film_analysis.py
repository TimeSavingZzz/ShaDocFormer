"""Generate paper figures from FiLM analysis data.

Input: film_analysis/ directory (output of analyze_film_collapse.py)
Output: PDF figures for paper

Figures:
  fig4_film_activations.pdf — tanh input/output histograms (SD7K vs RDD)
  fig5_gradient_flow.pdf — per-layer gradient norms comparison
  fig6_feature_pca.pdf — PCA of decoder features before/after FiLM
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


def plot_activation_histograms(data_dir, output_dir):
    """Fig 4: tanh activation saturation histograms (SD7K vs RDD)."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for col, dataset in enumerate(['sd7k', 'rdd']):
        path = os.path.join(data_dir, f'activation_{dataset}.json')
        if not os.path.exists(path):
            print(f"  [SKIP] {path} not found")
            continue
        with open(path) as f:
            data = json.load(f)

        # Find tanh-related stats
        tanh_keys = [k for k in data['forward'].keys() if 'tanh' in k.lower() and '_mean' in k]
        sat_keys = [k for k in data['forward'].keys() if 'sat_frac' in k]

        if tanh_keys:
            means = [data['forward'][k]['mean'] for k in tanh_keys]
            axes[0, col].bar(range(len(means)), means, alpha=0.7, color='steelblue')
            axes[0, col].set_title(f'{dataset.upper()} — tanh Input Mean')
            axes[0, col].set_ylabel('Mean Activation')
            axes[0, col].axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='sat. boundary')
            axes[0, col].axhline(y=-2.0, color='red', linestyle='--', alpha=0.5)

        if sat_keys:
            sat_vals = [data['forward'][k]['p95'] for k in sat_keys]
            axes[1, col].bar(range(len(sat_vals)), sat_vals, alpha=0.7, color='coral')
            axes[1, col].set_title(f'{dataset.upper()} — Saturation Fraction (p95)')
            axes[1, col].set_ylabel('Fraction in |x|>2.0')
            axes[1, col].set_ylim(0, 1)

    fig.suptitle('Figure 4: FiLM tanh Activation Analysis (SD7K vs RDD)', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig4_film_activations.pdf'))
    plt.close(fig)
    print("  -> fig4_film_activations.pdf")


def plot_gradient_flow(data_dir, output_dir):
    """Fig 5: Per-layer gradient norm comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for col, dataset in enumerate(['sd7k', 'rdd']):
        path = os.path.join(data_dir, f'gradient_{dataset}.json')
        if not os.path.exists(path):
            print(f"  [SKIP] {path} not found")
            continue
        with open(path) as f:
            data = json.load(f)

        grad_keys = sorted(data['gradient'].keys())
        names = [k.replace('_grad_norm', '').replace('sggf_dec', 'L') for k in grad_keys if '_grad_norm' in k]
        values = [data['gradient'][k]['mean'] for k in grad_keys if '_grad_norm' in k]

        if names:
            colors = ['red' if v > 100 else 'steelblue' for v in values]
            axes[col].barh(range(len(names)), values, color=colors, alpha=0.7)
            axes[col].set_yticks(range(len(names)))
            axes[col].set_yticklabels(names, fontsize=8)
            axes[col].set_xlabel('Mean Gradient Norm')
            axes[col].set_title(f'{dataset.upper()} — Per-Layer Gradient Norm')
            axes[col].axvline(x=100, color='red', linestyle='--', alpha=0.3, label='explosion threshold')

    fig.suptitle('Figure 5: FiLM Gradient Flow Analysis', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig5_gradient_flow.pdf'))
    plt.close(fig)
    print("  -> fig5_gradient_flow.pdf")


def plot_feature_pca(data_dir, output_dir):
    """Fig 6: PCA of decoder features before/after FiLM modulation."""
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for col, dataset in enumerate(['sd7k', 'rdd']):
        path = os.path.join(data_dir, f'features_{dataset}.npz')
        if not os.path.exists(path):
            print(f"  [SKIP] {path} not found")
            continue
        data = np.load(path, allow_pickle=True)

        # Find decoder and FiLM feature keys
        dec_keys = [k for k in data.keys() if 'decoder_level' in k.lower()]
        film_keys = [k for k in data.keys() if 'film' in k.lower()]

        all_feats = []
        labels = []
        for k in dec_keys[:2]:
            all_feats.append(data[k])
            labels.extend([f'{k}_dec'] * len(data[k]))
        for k in film_keys[:2]:
            all_feats.append(data[k])
            labels.extend([f'{k}_film'] * len(data[k]))

        if len(all_feats) < 2:
            continue

        X = np.concatenate(all_feats, axis=0)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        offset = 0
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_feats)))
        for i, feats in enumerate(all_feats):
            n = len(feats)
            axes[col].scatter(X_pca[offset:offset+n, 0], X_pca[offset:offset+n, 1],
                              c=[colors[i]], label=labels[i], alpha=0.6, s=20)
            offset += n

        axes[col].set_title(f'{dataset.upper()} — Feature PCA')
        axes[col].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[col].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        axes[col].legend(fontsize=7)

    fig.suptitle('Figure 6: Feature Space Analysis (Decoder vs FiLM-modulated)', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig6_feature_pca.pdf'))
    plt.close(fig)
    print("  -> fig6_feature_pca.pdf")


def plot_fisher_spectrum(data_dir, output_dir):
    """Supplementary: Fisher information eigenvalue spectrum."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for col, dataset in enumerate(['sd7k', 'rdd']):
        path = os.path.join(data_dir, f'fisher_{dataset}.json')
        if not os.path.exists(path):
            print(f"  [SKIP] {path} not found")
            continue
        with open(path) as f:
            data = json.load(f)

        values = sorted(data.values(), reverse=True)
        if values:
            axes[col].loglog(range(1, len(values)+1), values, 'o-', markersize=3, alpha=0.7)
            axes[col].set_title(f'{dataset.upper()} — Fisher Spectrum')
            axes[col].set_xlabel('Parameter Index (sorted)')
            axes[col].set_ylabel('Fisher Information (diag)')

    fig.suptitle('Supplementary: FiLM Fisher Information Spectrum', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'figS1_fisher_spectrum.pdf'))
    plt.close(fig)
    print("  -> figS1_fisher_spectrum.pdf")


def main():
    parser = argparse.ArgumentParser(description='Generate FiLM analysis figures')
    parser.add_argument('--data_dir', type=str, default='./film_analysis/',
                        help='Directory with analysis JSON/NPZ files')
    parser.add_argument('--output_dir', type=str, default='./film_analysis/',
                        help='Output directory for PDF figures')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Generating FiLM analysis figures...")
    plot_activation_histograms(args.data_dir, args.output_dir)
    plot_gradient_flow(args.data_dir, args.output_dir)
    plot_feature_pca(args.data_dir, args.output_dir)
    plot_fisher_spectrum(args.data_dir, args.output_dir)
    print(f"\nDone! Figures saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
