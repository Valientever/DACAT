#!/bin/bash

conda activate dacat

# Define log file location
LOG_FILE="/home/santhi/Documents/DACAT/src/Cholec80/results/baseline/log.txt"
touch $LOG_FILE

# Redirect all output (stdout & stderr) to log file and terminal
exec > >(tee -a "$LOG_FILE") 2>&1

# Add timestamp at the beginning of the log
echo "Logging started at $(date)"

export CUDA_VISIBLE_DEVICES=0

cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

# Function to log time for each step
log_time() {
    local step_name=$1
    local start_time=$SECONDS
    echo "[$(date)] - $step_name started."

    # Run the command
    shift
    "$@"

    # Check if the command failed
    if [ $? -ne 0 ]; then
        echo "[$(date)] - $step_name failed. Exiting."
        exit 1
    fi

    local end_time=$SECONDS
    local duration=$((end_time - start_time))
    echo "[$(date)] - $step_name completed successfully in ${duration} seconds."
}

# Step 1
log_time "Step 1" python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name baseline --step_1 phase_1 --step 1 --epochs 300 

# Step 2
log_time "Step 2" python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name baseline --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 300

# Step 3
log_time "Step 3" python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name baseline --step_1 phase_2 --step_3 predicts --step 3

# Step 4
cd /home/santhi/Documents/DACAT/src/Cholec80
log_time "Step 4" python3 eval.py --experiment_name baseline --predict_name 'predicts'

echo "All steps completed successfully."
