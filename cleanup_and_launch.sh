#!/bin/bash
# Self-contained cleanup + re-launch
cd /root/autodl-tmp/ShaDocFormer-main

# Kill ALL old training processes
pkill -9 -f "train_compare_models.py" 2>/dev/null
pkill -9 -f "run_ablation.sh" 2>/dev/null
sleep 3

# Verify GPU is clear
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# Clean any leftover checkpoint files from failed runs
rm -f experiment_results_rdd/full_200ep/textaware_v1_*.pth
rm -f experiment_results_rdd/full_200ep/textaware_v2_no_dtrm_*.pth
rm -f experiment_results_rdd/full_200ep/textaware_v2_fixed_*.pth
rm -f experiment_results_rdd/full_200ep/textaware_epoch*.pth
rm -f experiment_results_rdd/full_200ep/textaware_best.pth
rm -f experiment_results_rdd/full_200ep/textaware_final.pth
rm -f experiment_results_rdd/full_200ep/textaware_latest.pth
rm -f experiment_results_rdd/full_200ep/train_ablation_*.log

echo "Cleanup done, GPU memory: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

# Now launch ablation series
nohup bash run_ablation.sh > run_ablation_wrapper.log 2>&1 &
echo "Launched PID=$!"

sleep 5
# Check if training started
ps aux | grep train_compare | grep -v grep | head -3
echo "---"
head -10 experiment_results_rdd/full_200ep/train_ablation_v1.log 2>/dev/null || echo "Log not ready yet"
