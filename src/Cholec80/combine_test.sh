#!/usr/bin/env bash
set -eo pipefail  # Removed 'u' flag to avoid conda activation issues

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
# For whole-dataset aggregate evaluation (not per-video):
EVAL_SCRIPT="/home/santhi/Documents/DACAT/src/Cholec80/eval.py"
# For per-video evaluation, use: evaluate_per_video.py instead
STATS_SCRIPT="/home/santhi/Documents/DACAT/src/Cholec80/analyse_test.py"

BATCH_SIZE=1
NUM_WORKERS=4
ALPHA=0.05

# 3) Corruptions list
corruptions=(clean) # gnoise mblur dblur ueillumination seffect random)
predicts=(predicts r_predicts gn_predicts mb_predicts db_predicts ui_predicts se_predicts )
ext_predicts=(ext_r_predicts ext_gn_predicts ext_mb_predicts ext_db_predicts ext_ui_predicts ext_se_predicts )

# 4) Load checkpoints mapping (COMMENTED OUT - not needed for evaluation only)
# declare -A ckpt_map
# while IFS=": " read -r key path; do
#   ckpt_map["$key"]="$path"
# done < "$CKPTS_YAML"

#######################################################################################

# 5) Loop over corruptions and model tags
# for corr in "${corruptions[@]}"; do
#   for tag in baseline trained; do
#     # Determine YAML key (COMMENTED OUT - not needed for evaluation only)
#     # key="$corr"
#     # [[ "$tag" == "trained" ]] && key="${corr}_trained"
#     # ckpt="${ckpt_map[$key]:-}"

#     # if [[ -z "$ckpt" ]]; then
#     #   echo "⚠️  No checkpoint for '$key', skipping."
#     #   continue
#     # fi

#     # Construct experiment name based on corruption and tag
#     # e.g., "gaussian_noise_1210" or "random_1210" 
#     exp_name="${corr}_1210"
#     [[ "$tag" == "trained" ]] && exp_name="${corr}_1210"
    
#     # Predictions will be saved to: results/<exp_name>/predicts/
#     # preds_dir="$RESULTS_ROOT/$exp_name/ext_predicts"
#     preds_dir="$RESULTS_ROOT/$exp_name/$predicts"
#     mkdir -p "$preds_dir"

#     # Capture logs
#     log="$RESULTS_ROOT/$exp_name/ext_run.log"
#     exec > >(tee -a "$log") 2>&1

#     echo
#     echo "=============================================="
#     echo " corruption = $corr    model = $tag"
#     echo " experiment = $exp_name"
#     # echo " checkpoint = $ckpt"  # Not needed for evaluation only
#     echo "----------------------------------------------"

#     # 5.1) Generate predictions (COMMENTED OUT - predictions already exist)
#     # echo "[`date +%T`] Generating predictions..."
#     # python3 "$SAVE_SCRIPT" \
#     #   --resume "$ckpt" \
#     #   --task phase \
#     #   --corruption "$corr" \
#     #   --split "$SPLIT" \
#     #   --output_folder "$RESULTS_ROOT" \
#     #   --experiment_name "$exp_name" \
#     #   --step_3 "predicts"

#     # 5.2) Evaluate metrics (aggregate across all videos)
#     echo "[`date +%T`] Evaluating aggregate metrics..."
#     python3 "$EVAL_SCRIPT" \
#       --experiment_name "$exp_name" \
#       --predict_name "ext_predicts"
    
#     echo "[`date +%T`] Evaluation complete. Results in:"
#     echo "  - Predictions: $preds_dir/predv2_DACAT/"
#     echo "  - Ground truth: $preds_dir/gt/"
#     echo "  - Metrics: $preds_dir/ext_eval_results_total.txt"

#     echo "[`date +%T`] Done $corr / $tag"
#     echo
#   done
# done
##########################################################################################

for corr in "${corruptions[@]}"; do
  for tag in baseline trained; do
    # Determine YAML key (COMMENTED OUT - not needed for evaluation only)
    # key="$corr"
    # [[ "$tag" == "trained" ]] && key="${corr}_trained"
    # ckpt="${ckpt_map[$key]:-}"

    # if [[ -z "$ckpt" ]]; then
    #   echo "⚠️  No checkpoint for '$key', skipping."
    #   continue
    # fi

    # Construct experiment name based on corruption and tag
    # e.g., "gaussian_noise_1210" or "random_1210" 
    exp_name="${corr}_1210"
    [[ "$tag" == "trained" ]] && exp_name="${corr}_1210"
    # for predicts in "${predicts[@]}"; do
    for predicts in "${ext_predicts[@]}"; do
      # for predicts in "${predicts[@]}"; do
      
      # Predictions will be saved to: results/<exp_name>/predicts/
      # preds_dir="$RESULTS_ROOT/$exp_name/ext_predicts"
      preds_dir="$RESULTS_ROOT/$exp_name/$predicts"
      mkdir -p "$preds_dir"

      # Capture logs
      log="$RESULTS_ROOT/$exp_name/$predicts/run.log"
      exec > >(tee -a "$log") 2>&1

      echo
      echo "=============================================="
      echo " corruption = $corr    model = $tag"
      echo " experiment = $exp_name"
      # echo " checkpoint = $ckpt"  # Not needed for evaluation only
      echo "----------------------------------------------"

      # 5.1) Generate predictions (COMMENTED OUT - predictions already exist)
      # echo "[`date +%T`] Generating predictions..."
      # python3 "$SAVE_SCRIPT" \
      #   --resume "$ckpt" \
      #   --task phase \
      #   --corruption "$corr" \
      #   --split "$SPLIT" \
      #   --output_folder "$RESULTS_ROOT" \
      #   --experiment_name "$exp_name" \
      #   --step_3 "predicts"

      # 5.2) Evaluate metrics (aggregate across all videos)
      echo "[`date +%T`] Evaluating aggregate metrics..."
      python3 "$EVAL_SCRIPT" \
        --experiment_name "$exp_name" \
        --predict_name "$predicts"
      
      echo "[`date +%T`] Evaluation complete. Results in:"
      echo "  - Predictions: $preds_dir/predv2_DACAT/"
      echo "  - Ground truth: $preds_dir/gt/"
      echo "  - Metrics: $preds_dir/eval_results_total.txt"

      echo "[`date +%T`] Done $corr / $tag"
      echo
    done  # Close the predicts loop
  done  # Close the tag loop
done  # Close the corruptions loop

##########################################################################################

# 6) Aggregate all evaluation results (optional - for comparison)
echo
echo "=============================================="
echo " Aggregating evaluation results..."
echo "=============================================="
AGG_RESULTS="$RESULTS_ROOT/all_eval_results.txt"
echo "Experiment,Corruption,Model,Accuracy,Precision,Recall,Jaccard" > "$AGG_RESULTS"

for corr in "${corruptions[@]}"; do
  for tag in baseline trained; do
    for predicts in "${predicts[@]}"; do
      exp_name="${corr}_1210"
      [[ "$tag" == "trained" ]] && exp_name="${corr}_1210"

      eval_file="$RESULTS_ROOT/$exp_name/$predicts/eval_results_total.txt"
      if [[ -f "$eval_file" ]]; then
        # Extract metrics from eval_results_total.txt (customize based on actual format)
        echo "  ✓ Found: $exp_name with $predicts"
      fi
    done  # Close the predicts loop
  done  # Close the tag loop
done  # Close the corruptions loop
echo "✅ Aggregated results summary (if needed): $AGG_RESULTS"

# 7) Optional: Run statistical tests if you have a stats script
# Uncomment if you want to run statistical analysis
# echo
# echo "=============================================="
# echo " Running statistical tests (if applicable)..."
# echo "=============================================="
# python3 "$STATS_SCRIPT" \
#   --input "$AGG_RESULTS" \
#   --log   "$RESULTS_ROOT/statistical_log_whole.txt" \
#   --alpha "$ALPHA"

echo
echo "✅ Full pipeline complete!"
# echo "=============================================="
# echo " Results structure:"
# echo " • Each experiment in:        $RESULTS_ROOT/<corruption>_1210/$predicts/"
# echo " • Predictions in:            <exp>/$predicts/predv2_DACAT/"
# echo " • Ground truth in:           <exp>/$predicts/gt/"
# echo " • Evaluation results in:     <exp>/$predicts/eval_results_total.txt"
# echo " • Metrics CSV in:            <exp>/$predicts/metrics.csv"
# echo " • Visualizations in:         <exp>/$predicts/visualv2_DACAT/"
# echo "=============================================="

# echo "=============================================="
# echo " Results structure:"
# echo " • Each experiment in:        $RESULTS_ROOT/<corruption>_1210/"
# echo " • Predictions in:            <exp>/ext_predicts/predv2_DACAT/"
# echo " • Ground truth in:           <exp>/ext_predicts/gt/"
# echo " • Evaluation results in:     <exp>/ext_predicts/eval_results_total.txt"
# echo " • Metrics CSV in:            <exp>/ext_predicts/metrics.csv"
# echo " • Visualizations in:         <exp>/ext_predicts/visualv2_DACAT/"
# echo "=============================================="
