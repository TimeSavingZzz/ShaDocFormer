#!/bin/bash
# ShadowGuidedNAFNet 50-epoch verification
cd /root/autodl-tmp/ShaDocFormer-main
/root/miniconda3/envs/shadocformer/bin/python -u train_compare_models.py \
    --model shadow_guided \
    --epochs 50 \
    --batch_size 4 \
    --dataset rdd \
    --output "./experiment_results_rdd/full_200ep" \
    > "./experiment_results_rdd/full_200ep/train_shadow_guided_50ep.log" 2>&1
echo "shadow_guided done, exit=$?"

# Rename checkpoints
for f in ./experiment_results_rdd/full_200ep/shadow_guided_epoch*.pth \
         ./experiment_results_rdd/full_200ep/shadow_guided_best.pth \
         ./experiment_results_rdd/full_200ep/shadow_guided_final.pth \
         ./experiment_results_rdd/full_200ep/shadow_guided_latest.pth; do
    [ -f "$f" ] && mv "$f" "${f//shadow_guided/shadow_guided_50ep}"
done
echo "All done"
