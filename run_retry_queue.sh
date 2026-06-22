#!/bin/bash
# Retry: Concat (fixed) then Restormer (bs=1 to avoid OOM)
set -e
cd /root/autodl-tmp/ShaDocFormer-main
PY=/root/miniconda3/envs/shadocformer/bin/python
OUT=./experiment_results_rdd/full_200ep

echo "===== RETRY QUEUE START: $(date) ====="

# ---- [1/2] ShadowGuidedNAFNet Concat 200ep (fixed) ----
echo ">>> [1/2] Concat-only 200ep (fixed channels)"
$PY -u train_compare_models.py --model shadow_guided_concat --epochs 200 --batch_size 8 --dataset rdd --output "$OUT" > "$OUT/train_shadow_guided_concat_200ep.log" 2>&1
echo "concat done, exit=$?"
for f in "$OUT"/shadow_guided_concat_epoch*.pth "$OUT"/shadow_guided_concat_best.pth "$OUT"/shadow_guided_concat_final.pth "$OUT"/shadow_guided_concat_latest.pth; do
    [ -f "$f" ] && mv "$f" "${f//shadow_guided_concat/shadow_guided_concat_200ep}"
done

# ---- [2/2] Restormer 200ep (bs=1 to avoid OOM) ----
echo ">>> [2/2] Restormer 200ep (bs=1)"
$PY -u train_compare_models.py --model restormer --epochs 200 --batch_size 1 --dataset rdd --output "$OUT" > "$OUT/train_restormer_200ep.log" 2>&1
echo "restormer done, exit=$?"
for f in "$OUT"/restormer_epoch*.pth "$OUT"/restormer_best.pth "$OUT"/restormer_final.pth "$OUT"/restormer_latest.pth; do
    [ -f "$f" ] && mv "$f" "${f//restormer/restormer_200ep}"
done

echo "===== RETRY QUEUE END: $(date) ====="
