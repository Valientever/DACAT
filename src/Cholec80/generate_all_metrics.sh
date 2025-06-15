#!/usr/bin/env bash
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dacat

ROOT=/home/santhi/Documents/DACAT/src/Cholec80/results
EVAL=~/Documents/DACAT/src/Cholec80/evaluate_per_video.py

# Define training conditions
train_conditions=(
    "11_epoch_w28:clean"
    "gaussian_noise_w28:gaussian_noise"
    "motion_blur_w28:motion_blur"
    "defocus_blur_w28:defocus_blur"
    "smoke_effect_w28:smoke_effect"
    "uneven_illumination_w28:uneven_illumination"
    "random_w28:random_corruptions"
)

# Define evaluation predictions (checking what actually exists)
eval_predictions=(
    "cl_10_w28_predicts:clean"
    "gn_10_w28_predicts:gaussian_noise"
    "mb_10_w28_predicts:motion_blur"
    "db_10_w28_predicts:defocus_blur"
    "se_10_w28_predicts:smoke_effect"
    "ui_10_w28_predicts:uneven_illumination"
    "r_10_w28_predicts:random_corruptions"
    "predict_base:clean"
)

# Alternative naming (with typo)
eval_predictions_alt=(
    "gn_10_w28_predits:gaussian_noise"
    "mb_10_w28_predits:motion_blur"
    "db_10_w28_predits:defocus_blur"
    "se_10_w28_predits:smoke_effect"
    "ui_10_w28_predits:uneven_illumination"
    "r_10_w28_predits:random_corruptions"
)

echo "=== Generating All Cross-Corruption Combinations ==="
echo "Training conditions: ${#train_conditions[@]}"
echo "Evaluation predictions: ${#eval_predictions[@]} + ${#eval_predictions_alt[@]}"

# Generate all combinations
runs=()

for train_entry in "${train_conditions[@]}"; do
    IFS=':' read -r train_dir train_name <<< "$train_entry"
    
    # Check all evaluation predictions
    for eval_entry in "${eval_predictions[@]}" "${eval_predictions_alt[@]}"; do
        IFS=':' read -r eval_dir eval_name <<< "$eval_entry"
        
        # Check if this combination exists
        full_path="$ROOT/$train_dir/$eval_dir"
        if [[ -d "$full_path" ]]; then
            runs+=("$train_dir $eval_dir $train_name $eval_name")
            echo "✓ Found: $train_name -> $eval_name ($full_path)"
        fi
    done
done

echo ""
echo "=== Processing ${#runs[@]} combinations ==="

# Process each combination
for run in "${runs[@]}"; do
    read exp pred tc ec <<<"$run"
    outdir="$ROOT/$exp/$pred"
    outcsv="$outdir/metrics.csv"
    
    echo "Processing: $exp/$pred => train=$tc, eval=$ec"
    
    python3 "$EVAL" \
        --root_dir        "$ROOT" \
        --experiment_name "$exp" \
        --predict_name    "$pred" \
        --output_csv      "$outcsv" \
        --train_condition "$tc" \
        --eval_condition  "$ec"
done

# Aggregate all results
echo ""
echo "=== Aggregating Results ==="
agg="$ROOT/all_per_video_metrics.csv"
echo "video_id,train_condition,eval_condition,metric,score" > "$agg"

count=0
for run in "${runs[@]}"; do
    read exp pred tc ec <<<"$run"
    csvfile="$ROOT/$exp/$pred/metrics.csv"
    
    if [[ -f "$csvfile" ]]; then
        tail -n +2 "$csvfile" >> "$agg"
        ((count++))
        echo "  ✓ Added: $csvfile"
    else
        echo "  ✗ Missing: $csvfile"
    fi
done

echo ""
echo "=== Summary ==="
echo "Combined $count files into: $agg"
echo "Video range: 41-80"

# Show sample of results
echo ""
echo "=== Sample Results ==="
head -10 "$agg"

echo ""
echo "=== Combination Summary ==="
python3 -c "
import pandas as pd
df = pd.read_csv('$agg')
combinations = df[['train_condition', 'eval_condition']].drop_duplicates()
print(f'Total train-eval combinations: {len(combinations)}')
print('\\nCombinations:')
for _, row in combinations.sort_values(['train_condition', 'eval_condition']).iterrows():
    print(f'  {row.train_condition} -> {row.eval_condition}')
"
