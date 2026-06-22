#!/bin/bash
# Ablation experiments: 3 x 150 epochs
# v1 (MSER+gate+fixed L1) | v2 no-DTRM | v2 fixed-weights
cd /root/autodl-tmp/ShaDocFormer-main
PYTHON=/root/miniconda3/envs/shadocformer/bin/python
OUTDIR="./experiment_results_rdd/full_200ep"

echo "============================================================"
echo "[$(date)] ABLATION SERIES START"
echo "============================================================"

# ====== Exp 1: v1 baseline (MSER + gate + fixed L1) ======
echo "[$(date)] >>> ABLATION 1/3: v1 baseline (MSER+gate+fixed L1, 150ep)"
$PYTHON -u train_compare_models.py \
    --model textaware \
    --model_variant v1 \
    --epochs 150 \
    --batch_size 4 \
    --dataset rdd \
    --output "$OUTDIR" \
    > "$OUTDIR/train_ablation_v1.log" 2>&1
RC=$?
echo "[$(date)] <<< ABLATION 1/3 DONE (exit=$RC)"

# Rename checkpoints
for f in "$OUTDIR"/textaware_epoch*.pth "$OUTDIR"/textaware_best.pth "$OUTDIR"/textaware_final.pth "$OUTDIR"/textaware_latest.pth; do
    [ -f "$f" ] && mv "$f" "${f//textaware/textaware_v1}"
done

# ====== Exp 2: v2 no-DTRM (dtrm_weight=0) ======
echo "[$(date)] >>> ABLATION 2/3: v2 no-DTRM (dtrm_weight=0, 150ep)"
$PYTHON -u train_compare_models.py \
    --model textaware \
    --model_variant v2 \
    --epochs 150 \
    --batch_size 2 \
    --dtrm_weight 0 \
    --dataset rdd \
    --output "$OUTDIR" \
    > "$OUTDIR/train_ablation_v2_no_dtrm.log" 2>&1
RC=$?
echo "[$(date)] <<< ABLATION 2/3 DONE (exit=$RC)"

for f in "$OUTDIR"/textaware_epoch*.pth "$OUTDIR"/textaware_best.pth "$OUTDIR"/textaware_final.pth "$OUTDIR"/textaware_latest.pth; do
    [ -f "$f" ] && mv "$f" "${f//textaware/textaware_v2_no_dtrm}"
done

# ====== Exp 3: v2 fixed-weights (fixed_adaptive_weight=2.0) ======
echo "[$(date)] >>> ABLATION 3/3: v2 fixed-weights (fixed_adaptive=2.0, 150ep)"
$PYTHON -u train_compare_models.py \
    --model textaware \
    --model_variant v2 \
    --epochs 150 \
    --batch_size 2 \
    --fixed_adaptive_weight 2.0 \
    --dataset rdd \
    --output "$OUTDIR" \
    > "$OUTDIR/train_ablation_v2_fixed_weights.log" 2>&1
RC=$?
echo "[$(date)] <<< ABLATION 3/3 DONE (exit=$RC)"

for f in "$OUTDIR"/textaware_epoch*.pth "$OUTDIR"/textaware_best.pth "$OUTDIR"/textaware_final.pth "$OUTDIR"/textaware_latest.pth; do
    [ -f "$f" ] && mv "$f" "${f//textaware/textaware_v2_fixed_weights}"
done

echo "============================================================"
echo "[$(date)] ABLATION SERIES COMPLETE"
echo "============================================================"
