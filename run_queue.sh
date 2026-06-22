#!/bin/bash
# Master queue: ablations first, then comparison
# ShadowGuidedNAFNet ablations directly prove SGCA and ShadowEncoder contributions
set -e
cd /root/autodl-tmp/ShaDocFormer-main
PY=/root/miniconda3/envs/shadocformer/bin/python
OUT=./experiment_results_rdd/full_200ep

echo "===== QUEUE START: $(date) ====="

# ---- [1/3] ShadowGuidedNAFNet No-SGCA 200ep (ablation) ----
echo ">>> [1/3] No-SGCA ablation: concat fusion replaces channel attention"
$PY -u train_compare_models.py --model shadow_guided_no_sgca --epochs 200 --batch_size 4 --dataset rdd --output "$OUT" > "$OUT/train_shadow_guided_no_sgca_200ep.log" 2>&1
echo "no_sgca done, exit=$?"
for f in "$OUT"/shadow_guided_no_sgca_epoch*.pth "$OUT"/shadow_guided_no_sgca_best.pth "$OUT"/shadow_guided_no_sgca_final.pth "$OUT"/shadow_guided_no_sgca_latest.pth; do
    [ -f "$f" ] && mv "$f" "${f//shadow_guided_no_sgca/shadow_guided_no_sgca_200ep}"
done

# ---- [2/3] ShadowGuidedNAFNet Concat-only 200ep (ablation) ----
echo ">>> [2/3] Concat-only ablation: 4ch input, no ShadowEncoder, no SGCA"
$PY -u train_compare_models.py --model shadow_guided_concat --epochs 200 --batch_size 8 --dataset rdd --output "$OUT" > "$OUT/train_shadow_guided_concat_200ep.log" 2>&1
echo "concat done, exit=$?"
for f in "$OUT"/shadow_guided_concat_epoch*.pth "$OUT"/shadow_guided_concat_best.pth "$OUT"/shadow_guided_concat_final.pth "$OUT"/shadow_guided_concat_latest.pth; do
    [ -f "$f" ] && mv "$f" "${f//shadow_guided_concat/shadow_guided_concat_200ep}"
done

# ---- [3/3] Restormer 200ep (transformer comparison) ----
echo ">>> [3/3] Restormer 200ep (transformer baseline)"
$PY -u train_compare_models.py --model restormer --epochs 200 --batch_size 2 --dataset rdd --output "$OUT" > "$OUT/train_restormer_200ep.log" 2>&1
echo "restormer done, exit=$?"
for f in "$OUT"/restormer_epoch*.pth "$OUT"/restormer_best.pth "$OUT"/restormer_final.pth "$OUT"/restormer_latest.pth; do
    [ -f "$f" ] && mv "$f" "${f//restormer/restormer_200ep}"
done

echo "===== QUEUE END: $(date) ====="
