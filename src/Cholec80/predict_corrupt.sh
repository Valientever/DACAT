#!/usr/bin/env bash

# Activate your conda environment
conda activate dacat  # e.g., pytorch1_13

# Use GPU 0
export CUDA_VISIBLE_DEVICES=0

# Change directory to your train_scripts folder
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

# The first command-line argument to this script is the corruption mode (integer).
# If none is provided, default to 0 (no corruption).
CORRUPT_MODE=${1:-0}

echo "Running prediction with corruption mode = ${CORRUPT_MODE}"

# python3 save_predictions_onlinev2_longshort.py \
python3 save_predictions_onlinev2_longshort_corrupt.py \
    phase \
    --split cuhk \
    --backbone convnextv2 \
    --seq_len 1 \
    --resume /home/santhi/Documents/DACAT/checkpoints/Cholec80/checkpoint_best_acc.pth.tar \
    --corrupt ${CORRUPT_MODE}
