#!/bin/bash
cd /root/autodl-tmp/ShaDocFormer-main
/root/miniconda3/envs/shadocformer/bin/python -u train_compare_models.py \
    --model textaware \
    --model_variant v1 \
    --epochs 150 \
    --batch_size 4 \
    --dataset rdd \
    --output "./experiment_results_rdd/full_200ep" \
    > "./experiment_results_rdd/full_200ep/train_ablation_v1.log" 2>&1
echo "v1 retry done, exit=$?"
for f in ./experiment_results_rdd/full_200ep/textaware_epoch*.pth \
         ./experiment_results_rdd/full_200ep/textaware_best.pth \
         ./experiment_results_rdd/full_200ep/textaware_final.pth \
         ./experiment_results_rdd/full_200ep/textaware_latest.pth; do
    [ -f "$f" ] && mv "$f" "${f//textaware/textaware_v1}"
done
echo "v1 rename done"
