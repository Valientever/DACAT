#!/usr/bin/env bash
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dacat

ROOT=/home/santhi/Documents/DACAT/src/Cholec80/results_HeiChole
# os.mkdir -p "$ROOT"
EVAL=~/Documents/DACAT/src/Cholec80/evaluate_per_video.py

# Define (experiment,predict_name,train_cond,eval_cond) per run:
runs=(
    "11_epoch_w28 gn_10_w28_predicts clean gaussian_noise"
    "11_epoch_w28 mb_10_w28_predicts clean motion_blur"
    "11_epoch_w28 db_10_w28_predicts clean defocus_blur"
    "11_epoch_w28 ui_10_w28_predicts clean uneven_illumination"
    "11_epoch_w28 se_10_w28_predicts clean smoke_effect"
    "11_epoch_w28 r_10_w28_predicts clean random_corruptions"
    "gaussian_noise_w28 predict_base gaussian_noise gaussian_noise"
    "motion_blur_w28 predict_base motion_blur motion_blur"
    "defocus_blur_w28 predict_base defocus_blur defocus_blur"
    "uneven_illumination_w28 predict_base uneven_illumination uneven_illumination"
    "smoke_effect_w28 predict_base smoke_effect smoke_effect"
    "random_w28 predict_base random_corruptions random_corruptions"
  # … add one line per setting …
)

for run in "${runs[@]}"; do
  read exp pred tc ec <<<"$run"
  outdir="$ROOT/$exp/$pred"
  mkdir -p "$outdir"
  outcsv="$outdir/metrics.csv"

  echo "Running $exp / $pred ⇒ train_condition=$tc eval_condition=$ec"
  python3 "$EVAL" \
    --root_dir        "$ROOT" \
    --experiment_name "$exp" \
    --predict_name    "$pred" \
    --output_csv      "$outcsv" \
    --train_condition "$tc" \
    --eval_condition  "$ec"
done

# Finally, aggregate them:
agg="$ROOT/all_per_video_metrics.csv"
echo "video_id,train_condition,eval_condition,metric,score" > "$agg"
for f in $(printf "%s\n" "${runs[@]}" | awk '{print "'"$ROOT"'/" $1 "/" $2 "/metrics.csv"}'); do
  [[ -f "$f" ]] && tail -n +2 "$f" >> "$agg"
done

echo "Combined into $agg"
