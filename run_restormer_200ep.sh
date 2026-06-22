#!/bin/bash
cd /root/autodl-tmp/ShaDocFormer-main
/root/miniconda3/envs/shadocformer/bin/python -u train_compare_models.py \
    --model restormer \
    --epochs 200 \
    --batch_size 2 \
    --dataset rdd \
    --output "./experiment_results_rdd/full_200ep" \
    > "./experiment_results_rdd/full_200ep/train_restormer_200ep.log" 2>&1
echo "restormer 200ep done, exit=$?"
for f in ./experiment_results_rdd/full_200ep/restormer_epoch*.pth \
         ./experiment_results_rdd/full_200ep/restormer_best.pth \
         ./experiment_results_rdd/full_200ep/restormer_final.pth \
         ./experiment_results_rdd/full_200ep/restormer_latest.pth; do
    [ -f "$f" ] && mv "$f" "${f//restormer/restormer_200ep}"
done
echo "All done"
