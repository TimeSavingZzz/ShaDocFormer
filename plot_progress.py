"""
汇报用训练曲线: PSNR vs 实际训练时间
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 数据: Epoch → 实际耗时 (小时)
# ============================================================

textaware_time =  [0.4, 4.3, 8.2, 12.3, 16.3, 20.9]
textaware_psnr =  [23.49, 27.57, 28.89, 27.41, 30.82, 31.13]
textaware_tpsnr = [29.75, 31.69, 32.41, 32.47, 33.90, 34.08]
textaware_epoch = [1, 10, 20, 30, 40, 50]

baseline_time =  [0.4, 3.5, 7.0, 10.6, 14.2, 17.9]
baseline_psnr =  [18.92, 25.95, 28.35, 27.89, 30.09, 29.87]
baseline_tpsnr = [27.64, 31.06, 32.46, 31.59, 33.10, 33.12]
baseline_epoch = [1, 10, 20, 30, 40, 50]

# ============================================================
# Figure 1: PSNR vs 训练时间 + epoch 标注
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(baseline_time, baseline_psnr, 'o-', color='#2c7bb6', linewidth=2.5, markersize=9, label='ShaDocFormer Baseline')
ax.plot(textaware_time, textaware_psnr, 's--', color='#d7191c', linewidth=2.5, markersize=9, label='TextAware ShaDocFormer (Ours)')

# Epoch annotations
for t, p, e in zip(baseline_time, baseline_psnr, baseline_epoch):
    ax.annotate(f'E{e}', (t, p), textcoords="offset points", xytext=(5, 12), fontsize=9, color='#2c7bb6')
for t, p, e in zip(textaware_time, textaware_psnr, textaware_epoch):
    ax.annotate(f'E{e}', (t, p), textcoords="offset points", xytext=(5, -16), fontsize=9, color='#d7191c')

ax.set_xlabel('Training Time (hours)', fontsize=14)
ax.set_ylabel('PSNR (dB)', fontsize=14)
ax.set_title('PSNR vs Training Time — ShaDocFormer Baseline vs TextAware (RDD, 384x384, bs=4, AMP)', fontsize=14)
ax.legend(loc='lower right', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 22)

# Final value labels
ax.annotate(f'{baseline_psnr[-1]:.2f} dB', (baseline_time[-1], baseline_psnr[-1]),
            textcoords="offset points", xytext=(10, 0), fontsize=11, fontweight='bold', color='#2c7bb6')
ax.annotate(f'{textaware_psnr[-1]:.2f} dB', (textaware_time[-1], textaware_psnr[-1]),
            textcoords="offset points", xytext=(10, -20), fontsize=11, fontweight='bold', color='#d7191c')

plt.tight_layout()
fig.savefig('./experiment_results_rdd/psnr_vs_time.png', dpi=200)
plt.close()
print("Saved: psnr_vs_time.png")


# ============================================================
# Figure 2: Text-PSNR vs 训练时间
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(baseline_time, baseline_tpsnr, 'o-', color='#2c7bb6', linewidth=2.5, markersize=9, label='ShaDocFormer Baseline')
ax.plot(textaware_time, textaware_tpsnr, 's--', color='#d7191c', linewidth=2.5, markersize=9, label='TextAware ShaDocFormer (Ours)')

for t, p, e in zip(baseline_time, baseline_tpsnr, baseline_epoch):
    ax.annotate(f'E{e}', (t, p), textcoords="offset points", xytext=(5, 10), fontsize=9, color='#2c7bb6')
for t, p, e in zip(textaware_time, textaware_tpsnr, textaware_epoch):
    ax.annotate(f'E{e}', (t, p), textcoords="offset points", xytext=(5, -16), fontsize=9, color='#d7191c')

ax.set_xlabel('Training Time (hours)', fontsize=14)
ax.set_ylabel('Text-PSNR (dB)', fontsize=14)
ax.set_title('Text Region PSNR vs Training Time', fontsize=14)
ax.legend(loc='lower right', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 22)

ax.annotate(f'{baseline_tpsnr[-1]:.2f} dB', (baseline_time[-1], baseline_tpsnr[-1]),
            textcoords="offset points", xytext=(10, 0), fontsize=11, fontweight='bold', color='#2c7bb6')
ax.annotate(f'{textaware_tpsnr[-1]:.2f} dB', (textaware_time[-1], textaware_tpsnr[-1]),
            textcoords="offset points", xytext=(10, -20), fontsize=11, fontweight='bold', color='#d7191c')

plt.tight_layout()
fig.savefig('./experiment_results_rdd/textpsnr_vs_time.png', dpi=200)
plt.close()
print("Saved: textpsnr_vs_time.png")


# ============================================================
# Figure 3: 综合对比 (双Y: PSNR + TextPSNR, only TextAware)
# ============================================================
fig, ax1 = plt.subplots(figsize=(14, 6))

ax1.plot(textaware_time, textaware_psnr, 's-', color='#d7191c', linewidth=2.5, markersize=9, label='Overall PSNR')
ax1.set_xlabel('Training Time (hours)', fontsize=14)
ax1.set_ylabel('Overall PSNR (dB)', fontsize=14, color='#d7191c')
ax1.tick_params(axis='y', labelcolor='#d7191c')

ax2 = ax1.twinx()
ax2.plot(textaware_time, textaware_tpsnr, '^-', color='#2c7bb6', linewidth=2.5, markersize=9, label='Text Region PSNR')
ax2.set_ylabel('Text-PSNR (dB)', fontsize=14, color='#2c7bb6')
ax2.tick_params(axis='y', labelcolor='#2c7bb6')

# Epoch labels
for t, e in zip(textaware_time, textaware_epoch):
    ax1.axvline(x=t, color='gray', linestyle=':', alpha=0.4)
    ax1.annotate(f'Epoch {e}', (t, 23.5), fontsize=9, ha='center', color='gray')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=12)

ax1.grid(True, alpha=0.3)
ax1.set_title('TextAware ShaDocFormer: PSNR Progression (RDD, 384x384, bs=4, AMP, RTX 2080 Ti 22GB)', fontsize=14)

plt.tight_layout()
fig.savefig('./experiment_results_rdd/textaware_progress.png', dpi=200)
plt.close()
print("Saved: textaware_progress.png")

print("\nDone!")
