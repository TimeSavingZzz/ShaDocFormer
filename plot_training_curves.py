"""Plot 200-epoch training curves for all RDD models."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# ======== Data ========
nafnet = {
    'epoch': [1,10,20,30,50,100,150,200],
    'psnr': [22.86,26.66,28.80,28.32,30.71,32.16,33.06,33.34],
    'ssim': [0.6077,0.8444,0.8940,0.9070,0.9295,0.9476,0.9552,0.9570],
    'text_psnr': [31.33,32.10,34.36,32.77,34.81,36.05,36.74,36.93],
}

unet = {
    'epoch': [1,10,20,30,50,100,200],
    'psnr': [22.69,26.84,27.21,27.75,28.83,30.42,31.45],
    'ssim': [0.8219,0.8554,0.8556,0.8586,0.8599,0.8673,0.8687],
    'text_psnr': [29.95,32.78,33.97,33.45,35.08,35.87,36.42],
}

baseline = {
    'epoch': [1,10,20,30,50,100,150,200],
    'psnr': [22.41,25.52,27.45,27.69,29.05,29.24,31.53,31.91],
    'ssim': [0.8372,0.8593,0.8640,0.8652,0.8689,0.8709,0.8735,0.8740],
    'text_psnr': [29.31,30.37,32.50,31.26,32.65,33.19,34.77,35.12],
}

bedsr = {
    'epoch': [1,10,50,100,120,150,200],
    'psnr': [23.83,26.04,30.15,29.37,32.08,31.91,33.22],
    'ssim': [0.8330,0.8549,0.8575,0.8689,0.8711,0.8714,0.8748],
    'text_psnr': [29.55,29.62,31.88,33.41,33.89,34.20,34.41],
}

v2r1 = {
    'epoch': [1,30,60,100,150,200],
    'psnr': [23.36,30.67,31.93,32.03,34.07,34.56],
    'ssim': [0.8526,0.8712,0.8749,0.8757,0.8779,0.8784],
    'text_psnr': [0,0,0,0,0,0],
}

v2r3 = {
    'epoch': [1,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200],
    'psnr': [23.26,26.32,29.11,30.29,30.09,31.27,31.72,32.53,32.37,32.64,32.54,32.75,32.96,33.20,33.70,33.74,33.66,34.08,33.90,33.42,33.54],
    'ssim': [0.8518,0.8639,0.8696,0.8717,0.8718,0.8740,0.8746,0.8758,0.8755,0.8757,0.8761,0.8763,0.8770,0.8771,0.8778,0.8776,0.8773,0.8778,0.8776,0.8778,0.8779],
    'text_psnr': [24.16,27.66,30.08,31.97,31.50,32.51,33.05,33.73,33.60,34.19,34.04,34.66,34.83,34.74,35.42,35.55,35.34,35.70,35.58,34.75,35.17],
}

output_dir = './experiment_results_rdd/full_200ep'
os.makedirs(output_dir, exist_ok=True)

# ======== Style ========
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 13, 'axes.titlesize': 14,
    'legend.fontsize': 8, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

colors = {
    'NAFNet': '#2196F3', 'UNet': '#FF9800', 'baseline': '#4CAF50',
    'BEDSR': '#9C27B0', 'v2r1': '#9E9E9E', 'v2r3': '#E53935'
}
markers = {
    'NAFNet': 's', 'UNet': '^', 'baseline': 'v',
    'BEDSR': 'P', 'v2r1': 'o', 'v2r3': 'D'
}

# ======== Figure 1: PSNR ========
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(nafnet['epoch'], nafnet['psnr'], color=colors['NAFNet'], marker=markers['NAFNet'],
        markersize=5, lw=1.8, label='NAFNet')
ax.plot(unet['epoch'], unet['psnr'], color=colors['UNet'], marker=markers['UNet'],
        markersize=5, lw=1.8, label='UNet')
ax.plot(baseline['epoch'], baseline['psnr'], color=colors['baseline'], marker=markers['baseline'],
        markersize=5, lw=1.8, label='ShaDocFormer (baseline)')
ax.plot(bedsr['epoch'], bedsr['psnr'], color=colors['BEDSR'], marker=markers['BEDSR'],
        markersize=6, lw=1.8, label='BEDSR')
ax.plot(v2r1['epoch'], v2r1['psnr'], color=colors['v2r1'], marker=markers['v2r1'],
        markersize=5, lw=1.5, ls='--', label='TextAware v2 (LDTD collapse)')
ax.plot(v2r3['epoch'], v2r3['psnr'], color=colors['v2r3'], marker=markers['v2r3'],
        markersize=7, lw=2.2, label='TextAware v2 (ours)')
ax.set_xlabel('Epoch'); ax.set_ylabel('PSNR (dB)')
ax.set_title('RDD Shadow Removal — PSNR Comparison (384x384, 200 epochs)')
ax.legend(loc='lower right', ncol=2); ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'training_curve_psnr.png'))
plt.close(fig)

# ======== Figure 2: SSIM ========
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(nafnet['epoch'], nafnet['ssim'], color=colors['NAFNet'], marker=markers['NAFNet'],
        markersize=5, lw=1.8, label='NAFNet')
ax.plot(unet['epoch'], unet['ssim'], color=colors['UNet'], marker=markers['UNet'],
        markersize=5, lw=1.8, label='UNet')
ax.plot(baseline['epoch'], baseline['ssim'], color=colors['baseline'], marker=markers['baseline'],
        markersize=5, lw=1.8, label='ShaDocFormer (baseline)')
ax.plot(bedsr['epoch'], bedsr['ssim'], color=colors['BEDSR'], marker=markers['BEDSR'],
        markersize=6, lw=1.8, label='BEDSR')
ax.plot(v2r1['epoch'], v2r1['ssim'], color=colors['v2r1'], marker=markers['v2r1'],
        markersize=5, lw=1.5, ls='--', label='TextAware v2 (LDTD collapse)')
ax.plot(v2r3['epoch'], v2r3['ssim'], color=colors['v2r3'], marker=markers['v2r3'],
        markersize=7, lw=2.2, label='TextAware v2 (ours)')
ax.set_xlabel('Epoch'); ax.set_ylabel('SSIM')
ax.set_title('RDD Shadow Removal — SSIM Comparison (384x384, 200 epochs)')
ax.legend(loc='lower right', ncol=2); ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'training_curve_ssim.png'))
plt.close(fig)

# ======== Figure 3: Text-PSNR ========
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(nafnet['epoch'], nafnet['text_psnr'], color=colors['NAFNet'], marker=markers['NAFNet'],
        markersize=5, lw=1.8, label='NAFNet')
ax.plot(unet['epoch'], unet['text_psnr'], color=colors['UNet'], marker=markers['UNet'],
        markersize=5, lw=1.8, label='UNet')
ax.plot(baseline['epoch'], baseline['text_psnr'], color=colors['baseline'], marker=markers['baseline'],
        markersize=5, lw=1.8, label='ShaDocFormer (baseline)')
ax.plot(bedsr['epoch'], bedsr['text_psnr'], color=colors['BEDSR'], marker=markers['BEDSR'],
        markersize=6, lw=1.8, label='BEDSR')
ax.plot(v2r3['epoch'], v2r3['text_psnr'], color=colors['v2r3'], marker=markers['v2r3'],
        markersize=7, lw=2.2, label='TextAware v2 (ours)')
ax.set_xlabel('Epoch'); ax.set_ylabel('Text-PSNR (dB)')
ax.set_title('RDD Shadow Removal — Text-PSNR Comparison (384x384, 200 epochs)')
ax.legend(loc='lower right', ncol=2); ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
ann_text = 'v2 (collapse): TextPSNR=0\n\n(gradient conflict)'
ax.annotate(ann_text, xy=(100, 5), fontsize=8, color='#999', ha='center',
            bbox=dict(boxstyle='round', facecolor='#f5f5f5', edgecolor='#ccc'))
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'training_curve_textpsnr.png'))
plt.close(fig)

# ======== Figure 4: Combined overview (3 subplots) ========
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

model_data = [
    ('NAFNet', nafnet, colors['NAFNet'], markers['NAFNet'], '-'),
    ('UNet', unet, colors['UNet'], markers['UNet'], '-'),
    ('baseline', baseline, colors['baseline'], markers['baseline'], '-'),
    ('BEDSR', bedsr, colors['BEDSR'], markers['BEDSR'], '-'),
    ('v2 (collapse)', v2r1, colors['v2r1'], markers['v2r1'], '--'),
    ('v2 (ours)', v2r3, colors['v2r3'], markers['v2r3'], '-'),
]

for ax, metric, ylabel, title in [
    (axes[0], 'psnr', 'PSNR (dB)', 'PSNR'),
    (axes[1], 'ssim', 'SSIM', 'SSIM'),
    (axes[2], 'text_psnr', 'Text-PSNR (dB)', 'Text-PSNR')]:
    for name, data, c, m, ls in model_data:
        ms = 7 if name == 'v2 (ours)' else 4
        lw_val = 2.2 if name == 'v2 (ours)' else 1.5
        if metric == 'text_psnr' and name == 'v2 (collapse)':
            continue  # skip TextPSNR=0 for collapsed version
        ax.plot(data['epoch'], data[metric], color=c, marker=m,
                markersize=ms, lw=lw_val, ls=ls, label=name)
    ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

fig.suptitle('ShaDocFormer v2 — RDD 200-Epoch Full Comparison', fontweight='bold', fontsize=16)
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'training_curve_overview.png'))
plt.close(fig)

# ======== Console Summary ========
print("Saved 4 figures to", output_dir)
print("  training_curve_psnr.png")
print("  training_curve_ssim.png")
print("  training_curve_textpsnr.png")
print("  training_curve_overview.png")
print()
print(f"{'Model':<25} {'PSNR':>8} {'SSIM':>8} {'TextPSNR':>10}")
print("-" * 55)
print(f"{'NAFNet (best E190)':<25} {max(nafnet['psnr']):>8.2f} {max(nafnet['ssim']):>8.4f} {max(nafnet['text_psnr']):>10.2f}")
print(f"{'BEDSR (E200)':<25} {bedsr['psnr'][-1]:>8.2f} {bedsr['ssim'][-1]:>8.4f} {bedsr['text_psnr'][-1]:>10.2f}")
print(f"{'v2 ours (peak E170)':<25} {max(v2r3['psnr']):>8.2f} {v2r3['ssim'][v2r3['psnr'].index(max(v2r3['psnr']))]:>8.4f} {v2r3['text_psnr'][v2r3['psnr'].index(max(v2r3['psnr']))]:>10.2f}")
print(f"{'v2 ours (final E200)':<25} {v2r3['psnr'][-1]:>8.2f} {v2r3['ssim'][-1]:>8.4f} {v2r3['text_psnr'][-1]:>10.2f}")
print(f"{'v2 collapse (E200)':<25} {v2r1['psnr'][-1]:>8.2f} {v2r1['ssim'][-1]:>8.4f} {'0.00':>10}")
print(f"{'baseline (E200)':<25} {baseline['psnr'][-1]:>8.2f} {baseline['ssim'][-1]:>8.4f} {baseline['text_psnr'][-1]:>10.2f}")
print(f"{'UNet (E200)':<25} {unet['psnr'][-1]:>8.2f} {unet['ssim'][-1]:>8.4f} {unet['text_psnr'][-1]:>10.2f}")
