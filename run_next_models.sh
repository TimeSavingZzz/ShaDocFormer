#!/bin/bash
# Auto-continue: after textaware v2 finishes, run remaining models
cd /root/autodl-tmp/ShaDocFormer-main

echo "[$(date)] Waiting for textaware v2 to finish..."
while pgrep -f "train_compare_models.py.*textaware" > /dev/null 2>&1; do
    sleep 60
done

echo "[$(date)] v2 done, starting docdeshadower + bedsr + restormer"
/root/miniconda3/envs/shadocformer/bin/python -u train_compare_models.py \
    --model docdeshadower,bedsr,restormer \
    --epochs 200 \
    --batch_size 4 \
    --dataset rdd \
    --output ./experiment_results_rdd/full_200ep/ \
    --resume auto \
    > train_phase3.log 2>&1

echo "[$(date)] All models done"
