#!/usr/bin/env bash
set -euo pipefail

# 1) Env setup
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dacat
export CUDA_VISIBLE_DEVICES=0

# 2) Configuration
CKPTS_YAML="/home/santhi/Documents/DACAT/src/Cholec80/results/ckpts.yaml"
DATA_ROOT="/home/santhi/Documents/DACAT/src/Cholec80/data"
SPLIT="cuhk"
RESULTS_ROOT="/home/santhi/Documents/DACAT/src/Cholec80/results"

SAVE_SCRIPT="/home/santhi/Documents/DACAT/src/Cholec80/train_scripts/save_predictions_onlinev2_longshort.py"
EVAL_SCRIPT="/home/santhi/Documents/DACAT/src/Cholec80/train_scripts/eval.py"
STATS_SCRIPT="/home/santhi/Documents/DACAT/src/Cholec80/analyse_test.py"

BATCH_SIZE=1
NUM_WORKERS=4
ALPHA=0.05

# 3) Corruptions list
corruptions=(gaussian_noise motion_blur defocus_blur uneven_illumination smoke_effect random_corruption)

# 4) Load checkpoints mapping
declare -A ckpt_map
while IFS=": " read -r key path; do
  ckpt_map["$key"]="$path"
done < "$CKPTS_YAML"

# 5) Loop over corruptions and model tags
for corr in "${corruptions[@]}"; do
  for tag in baseline trained; do
    # Determine YAML key
    key="$corr"
    [[ "$tag" == "trained" ]] && key="${corr}_trained"
    ckpt="${ckpt_map[$key]:-}"

    if [[ -z "$ckpt" ]]; then
      echo "⚠️  No checkpoint for '$key', skipping."
      continue
    fi

    out="$RESULTS_ROOT/$corr/$tag"
    preds_dir="$out/predictions"
    metrics_dir="$out/metrics"
    mkdir -p "$preds_dir" "$metrics_dir"

    # Capture logs
    log="$out/run.log"
    exec > >(tee -a "$log") 2>&1

    echo
    echo "=============================================="
    echo " corruption = $corr    model = $tag"
    echo " checkpoint = $ckpt"
    echo "----------------------------------------------"

    # 5.1) Generate predictions
    echo "[`date +%T`] Generating predictions..."
    python3 "$SAVE_SCRIPT" \
      --resume "$ckpt" \
      --phase "phase" \
      --corruption "$corr" \
      --experiment_name "$tag" \
      --predict_name predictions \
      --output_folder "$preds_dir"

    # 5.2) Evaluate per-video metrics
    echo "[`date +%T`] Evaluating per-video metrics..."
    python3 "$EVAL_SCRIPT" \
      --root_dir       "$RESULTS_ROOT/$corr" \
      --experiment_name "$tag" \
      --predict_name   "predictions" \
      --output_csv     "$metrics_dir/metrics.csv"

    echo "[`date +%T`] Done $corr / $tag"
    echo
  done
done

# 6) Aggregate all per-video metrics
AGG="$RESULTS_ROOT/scores.csv"
echo "video_id,train_condition,eval_condition,metric,score" > "$AGG"
for f in "$RESULTS_ROOT"/*/{baseline,trained}/metrics.csv; do
  [[ -f "$f" ]] && tail -n+2 "$f" >> "$AGG"
done
echo "🗄  Aggregated metrics ⇒ $AGG"

# 7) Run paired‐Wilcoxon tests
echo
echo "=============================================="
echo " Running statistical tests (Wilcoxon)..."
echo "=============================================="
python3 "$STATS_SCRIPT" \
  --input "$AGG" \
  --log   "$RESULTS_ROOT/statistical_log.txt" \
  --alpha "$ALPHA"

echo
echo "✅ Full pipeline complete!"
echo " • Predictions & logs in:     $RESULTS_ROOT/<corruption>/{baseline,trained}/"
echo " • Per-video CSVs in:         $RESULTS_ROOT/<corruption>/{baseline,trained}/metrics.csv"
echo " • Aggregated scores:         $AGG"
echo " • Stats log:                 $RESULTS_ROOT/statistical_log.txt"
