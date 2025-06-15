#!/bin/bash
# Cross-corruption experiment commands

# Experiment 1: clean → gaussian_noise
echo 'Starting experiment 1: clean → gaussian_noise'
export EXPERIMENT_NAME='cross_clean_to_gaussian_noise'
export TRAIN_CORRUPTION=''
export EVAL_CORRUPTION='gaussian_noise'

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption gaussian_noise

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 1: clean → gaussian_noise'
sleep 10  # Brief pause between experiments

# Experiment 2: clean → motion_blur
echo 'Starting experiment 2: clean → motion_blur'
export EXPERIMENT_NAME='cross_clean_to_motion_blur'
export TRAIN_CORRUPTION=''
export EVAL_CORRUPTION='motion_blur'

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption motion_blur

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 2: clean → motion_blur'
sleep 10  # Brief pause between experiments

# Experiment 3: clean → defocus_blur
echo 'Starting experiment 3: clean → defocus_blur'
export EXPERIMENT_NAME='cross_clean_to_defocus_blur'
export TRAIN_CORRUPTION=''
export EVAL_CORRUPTION='defocus_blur'

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption defocus_blur

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 3: clean → defocus_blur'
sleep 10  # Brief pause between experiments

# Experiment 4: clean → uneven_illumination
echo 'Starting experiment 4: clean → uneven_illumination'
export EXPERIMENT_NAME='cross_clean_to_uneven_illumination'
export TRAIN_CORRUPTION=''
export EVAL_CORRUPTION='uneven_illumination'

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption uneven_illumination

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 4: clean → uneven_illumination'
sleep 10  # Brief pause between experiments

# Experiment 5: clean → smoke_effect
echo 'Starting experiment 5: clean → smoke_effect'
export EXPERIMENT_NAME='cross_clean_to_smoke_effect'
export TRAIN_CORRUPTION=''
export EVAL_CORRUPTION='smoke_effect'

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption smoke_effect

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 5: clean → smoke_effect'
sleep 10  # Brief pause between experiments

# Experiment 6: clean → random
echo 'Starting experiment 6: clean → random'
export EXPERIMENT_NAME='cross_clean_to_random'
export TRAIN_CORRUPTION=''
export EVAL_CORRUPTION='random'

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption random

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 6: clean → random'
sleep 10  # Brief pause between experiments

# Experiment 7: gaussian_noise → clean
echo 'Starting experiment 7: gaussian_noise → clean'
export EXPERIMENT_NAME='cross_gaussian_noise_to_clean'
export TRAIN_CORRUPTION='gaussian_noise'
export EVAL_CORRUPTION=''

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 --corruption gaussian_noise

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 --corruption gaussian_noise

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 7: gaussian_noise → clean'
sleep 10  # Brief pause between experiments

# Experiment 8: gaussian_noise → motion_blur
echo 'Starting experiment 8: gaussian_noise → motion_blur'
export EXPERIMENT_NAME='cross_gaussian_noise_to_motion_blur'
export TRAIN_CORRUPTION='gaussian_noise'
export EVAL_CORRUPTION='motion_blur'

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 --corruption gaussian_noise

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 --corruption gaussian_noise

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption motion_blur

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 8: gaussian_noise → motion_blur'
sleep 10  # Brief pause between experiments

# Experiment 9: gaussian_noise → defocus_blur
echo 'Starting experiment 9: gaussian_noise → defocus_blur'
export EXPERIMENT_NAME='cross_gaussian_noise_to_defocus_blur'
export TRAIN_CORRUPTION='gaussian_noise'
export EVAL_CORRUPTION='defocus_blur'

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 --corruption gaussian_noise

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 --corruption gaussian_noise

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption defocus_blur

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 9: gaussian_noise → defocus_blur'
sleep 10  # Brief pause between experiments

# Experiment 10: gaussian_noise → uneven_illumination
echo 'Starting experiment 10: gaussian_noise → uneven_illumination'
export EXPERIMENT_NAME='cross_gaussian_noise_to_uneven_illumination'
export TRAIN_CORRUPTION='gaussian_noise'
export EVAL_CORRUPTION='uneven_illumination'

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 --corruption gaussian_noise

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 --corruption gaussian_noise

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption uneven_illumination

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 10: gaussian_noise → uneven_illumination'
sleep 10  # Brief pause between experiments

# Experiment 11: gaussian_noise → smoke_effect
echo 'Starting experiment 11: gaussian_noise → smoke_effect'
export EXPERIMENT_NAME='cross_gaussian_noise_to_smoke_effect'
export TRAIN_CORRUPTION='gaussian_noise'
export EVAL_CORRUPTION='smoke_effect'

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 --corruption gaussian_noise

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 --corruption gaussian_noise

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption smoke_effect

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 11: gaussian_noise → smoke_effect'
sleep 10  # Brief pause between experiments

# Experiment 12: gaussian_noise → random
echo 'Starting experiment 12: gaussian_noise → random'
export EXPERIMENT_NAME='cross_gaussian_noise_to_random'
export TRAIN_CORRUPTION='gaussian_noise'
export EVAL_CORRUPTION='random'

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 --corruption gaussian_noise

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 --corruption gaussian_noise

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption random

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 12: gaussian_noise → random'
sleep 10  # Brief pause between experiments

# Experiment 13: motion_blur → clean
echo 'Starting experiment 13: motion_blur → clean'
export EXPERIMENT_NAME='cross_motion_blur_to_clean'
export TRAIN_CORRUPTION='motion_blur'
export EVAL_CORRUPTION=''

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 --corruption motion_blur

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 --corruption motion_blur

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 13: motion_blur → clean'
sleep 10  # Brief pause between experiments

# Experiment 14: motion_blur → gaussian_noise
echo 'Starting experiment 14: motion_blur → gaussian_noise'
export EXPERIMENT_NAME='cross_motion_blur_to_gaussian_noise'
export TRAIN_CORRUPTION='motion_blur'
export EVAL_CORRUPTION='gaussian_noise'

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 --corruption motion_blur

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 --corruption motion_blur

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption gaussian_noise

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 14: motion_blur → gaussian_noise'
sleep 10  # Brief pause between experiments

# Experiment 15: motion_blur → defocus_blur
echo 'Starting experiment 15: motion_blur → defocus_blur'
export EXPERIMENT_NAME='cross_motion_blur_to_defocus_blur'
export TRAIN_CORRUPTION='motion_blur'
export EVAL_CORRUPTION='defocus_blur'

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 --corruption motion_blur

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 --corruption motion_blur

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption defocus_blur

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 15: motion_blur → defocus_blur'
sleep 10  # Brief pause between experiments

# Experiment 16: motion_blur → uneven_illumination
echo 'Starting experiment 16: motion_blur → uneven_illumination'
export EXPERIMENT_NAME='cross_motion_blur_to_uneven_illumination'
export TRAIN_CORRUPTION='motion_blur'
export EVAL_CORRUPTION='uneven_illumination'

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 --corruption motion_blur

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 --corruption motion_blur

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption uneven_illumination

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 16: motion_blur → uneven_illumination'
sleep 10  # Brief pause between experiments

# Experiment 17: motion_blur → smoke_effect
echo 'Starting experiment 17: motion_blur → smoke_effect'
export EXPERIMENT_NAME='cross_motion_blur_to_smoke_effect'
export TRAIN_CORRUPTION='motion_blur'
export EVAL_CORRUPTION='smoke_effect'

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 --corruption motion_blur

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 --corruption motion_blur

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption smoke_effect

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 17: motion_blur → smoke_effect'
sleep 10  # Brief pause between experiments

# Experiment 18: motion_blur → random
echo 'Starting experiment 18: motion_blur → random'
export EXPERIMENT_NAME='cross_motion_blur_to_random'
export TRAIN_CORRUPTION='motion_blur'
export EVAL_CORRUPTION='random'

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 --corruption motion_blur

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 --corruption motion_blur

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption random

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 18: motion_blur → random'
sleep 10  # Brief pause between experiments

# Experiment 19: defocus_blur → clean
echo 'Starting experiment 19: defocus_blur → clean'
export EXPERIMENT_NAME='cross_defocus_blur_to_clean'
export TRAIN_CORRUPTION='defocus_blur'
export EVAL_CORRUPTION=''

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 --corruption defocus_blur

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 --corruption defocus_blur

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 19: defocus_blur → clean'
sleep 10  # Brief pause between experiments

# Experiment 20: defocus_blur → gaussian_noise
echo 'Starting experiment 20: defocus_blur → gaussian_noise'
export EXPERIMENT_NAME='cross_defocus_blur_to_gaussian_noise'
export TRAIN_CORRUPTION='defocus_blur'
export EVAL_CORRUPTION='gaussian_noise'

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 --corruption defocus_blur

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 --corruption defocus_blur

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption gaussian_noise

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 20: defocus_blur → gaussian_noise'
sleep 10  # Brief pause between experiments

# Experiment 21: defocus_blur → motion_blur
echo 'Starting experiment 21: defocus_blur → motion_blur'
export EXPERIMENT_NAME='cross_defocus_blur_to_motion_blur'
export TRAIN_CORRUPTION='defocus_blur'
export EVAL_CORRUPTION='motion_blur'

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 --corruption defocus_blur

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 --corruption defocus_blur

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption motion_blur

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 21: defocus_blur → motion_blur'
sleep 10  # Brief pause between experiments

# Experiment 22: defocus_blur → uneven_illumination
echo 'Starting experiment 22: defocus_blur → uneven_illumination'
export EXPERIMENT_NAME='cross_defocus_blur_to_uneven_illumination'
export TRAIN_CORRUPTION='defocus_blur'
export EVAL_CORRUPTION='uneven_illumination'

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 --corruption defocus_blur

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 --corruption defocus_blur

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption uneven_illumination

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 22: defocus_blur → uneven_illumination'
sleep 10  # Brief pause between experiments

# Experiment 23: defocus_blur → smoke_effect
echo 'Starting experiment 23: defocus_blur → smoke_effect'
export EXPERIMENT_NAME='cross_defocus_blur_to_smoke_effect'
export TRAIN_CORRUPTION='defocus_blur'
export EVAL_CORRUPTION='smoke_effect'

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 --corruption defocus_blur

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 --corruption defocus_blur

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption smoke_effect

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 23: defocus_blur → smoke_effect'
sleep 10  # Brief pause between experiments

# Experiment 24: defocus_blur → random
echo 'Starting experiment 24: defocus_blur → random'
export EXPERIMENT_NAME='cross_defocus_blur_to_random'
export TRAIN_CORRUPTION='defocus_blur'
export EVAL_CORRUPTION='random'

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 --corruption defocus_blur

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 --corruption defocus_blur

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption random

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 24: defocus_blur → random'
sleep 10  # Brief pause between experiments

# Experiment 25: uneven_illumination → clean
echo 'Starting experiment 25: uneven_illumination → clean'
export EXPERIMENT_NAME='cross_uneven_illumination_to_clean'
export TRAIN_CORRUPTION='uneven_illumination'
export EVAL_CORRUPTION=''

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 --corruption uneven_illumination

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 --corruption uneven_illumination

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 25: uneven_illumination → clean'
sleep 10  # Brief pause between experiments

# Experiment 26: uneven_illumination → gaussian_noise
echo 'Starting experiment 26: uneven_illumination → gaussian_noise'
export EXPERIMENT_NAME='cross_uneven_illumination_to_gaussian_noise'
export TRAIN_CORRUPTION='uneven_illumination'
export EVAL_CORRUPTION='gaussian_noise'

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 --corruption uneven_illumination

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 --corruption uneven_illumination

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption gaussian_noise

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 26: uneven_illumination → gaussian_noise'
sleep 10  # Brief pause between experiments

# Experiment 27: uneven_illumination → motion_blur
echo 'Starting experiment 27: uneven_illumination → motion_blur'
export EXPERIMENT_NAME='cross_uneven_illumination_to_motion_blur'
export TRAIN_CORRUPTION='uneven_illumination'
export EVAL_CORRUPTION='motion_blur'

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 --corruption uneven_illumination

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 --corruption uneven_illumination

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption motion_blur

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 27: uneven_illumination → motion_blur'
sleep 10  # Brief pause between experiments

# Experiment 28: uneven_illumination → defocus_blur
echo 'Starting experiment 28: uneven_illumination → defocus_blur'
export EXPERIMENT_NAME='cross_uneven_illumination_to_defocus_blur'
export TRAIN_CORRUPTION='uneven_illumination'
export EVAL_CORRUPTION='defocus_blur'

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 --corruption uneven_illumination

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 --corruption uneven_illumination

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption defocus_blur

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 28: uneven_illumination → defocus_blur'
sleep 10  # Brief pause between experiments

# Experiment 29: uneven_illumination → smoke_effect
echo 'Starting experiment 29: uneven_illumination → smoke_effect'
export EXPERIMENT_NAME='cross_uneven_illumination_to_smoke_effect'
export TRAIN_CORRUPTION='uneven_illumination'
export EVAL_CORRUPTION='smoke_effect'

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 --corruption uneven_illumination

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 --corruption uneven_illumination

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption smoke_effect

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 29: uneven_illumination → smoke_effect'
sleep 10  # Brief pause between experiments

# Experiment 30: uneven_illumination → random
echo 'Starting experiment 30: uneven_illumination → random'
export EXPERIMENT_NAME='cross_uneven_illumination_to_random'
export TRAIN_CORRUPTION='uneven_illumination'
export EVAL_CORRUPTION='random'

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 --corruption uneven_illumination

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 --corruption uneven_illumination

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption random

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 30: uneven_illumination → random'
sleep 10  # Brief pause between experiments

# Experiment 31: smoke_effect → clean
echo 'Starting experiment 31: smoke_effect → clean'
export EXPERIMENT_NAME='cross_smoke_effect_to_clean'
export TRAIN_CORRUPTION='smoke_effect'
export EVAL_CORRUPTION=''

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 --corruption smoke_effect

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 --corruption smoke_effect

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 31: smoke_effect → clean'
sleep 10  # Brief pause between experiments

# Experiment 32: smoke_effect → gaussian_noise
echo 'Starting experiment 32: smoke_effect → gaussian_noise'
export EXPERIMENT_NAME='cross_smoke_effect_to_gaussian_noise'
export TRAIN_CORRUPTION='smoke_effect'
export EVAL_CORRUPTION='gaussian_noise'

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 --corruption smoke_effect

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 --corruption smoke_effect

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption gaussian_noise

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 32: smoke_effect → gaussian_noise'
sleep 10  # Brief pause between experiments

# Experiment 33: smoke_effect → motion_blur
echo 'Starting experiment 33: smoke_effect → motion_blur'
export EXPERIMENT_NAME='cross_smoke_effect_to_motion_blur'
export TRAIN_CORRUPTION='smoke_effect'
export EVAL_CORRUPTION='motion_blur'

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 --corruption smoke_effect

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 --corruption smoke_effect

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption motion_blur

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 33: smoke_effect → motion_blur'
sleep 10  # Brief pause between experiments

# Experiment 34: smoke_effect → defocus_blur
echo 'Starting experiment 34: smoke_effect → defocus_blur'
export EXPERIMENT_NAME='cross_smoke_effect_to_defocus_blur'
export TRAIN_CORRUPTION='smoke_effect'
export EVAL_CORRUPTION='defocus_blur'

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 --corruption smoke_effect

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 --corruption smoke_effect

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption defocus_blur

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 34: smoke_effect → defocus_blur'
sleep 10  # Brief pause between experiments

# Experiment 35: smoke_effect → uneven_illumination
echo 'Starting experiment 35: smoke_effect → uneven_illumination'
export EXPERIMENT_NAME='cross_smoke_effect_to_uneven_illumination'
export TRAIN_CORRUPTION='smoke_effect'
export EVAL_CORRUPTION='uneven_illumination'

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 --corruption smoke_effect

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 --corruption smoke_effect

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption uneven_illumination

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 35: smoke_effect → uneven_illumination'
sleep 10  # Brief pause between experiments

# Experiment 36: smoke_effect → random
echo 'Starting experiment 36: smoke_effect → random'
export EXPERIMENT_NAME='cross_smoke_effect_to_random'
export TRAIN_CORRUPTION='smoke_effect'
export EVAL_CORRUPTION='random'

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 --corruption smoke_effect

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 --corruption smoke_effect

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption random

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 36: smoke_effect → random'
sleep 10  # Brief pause between experiments

# Experiment 37: random → clean
echo 'Starting experiment 37: random → clean'
export EXPERIMENT_NAME='cross_random_to_clean'
export TRAIN_CORRUPTION='random'
export EVAL_CORRUPTION=''

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 --corruption random

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 --corruption random

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 37: random → clean'
sleep 10  # Brief pause between experiments

# Experiment 38: random → gaussian_noise
echo 'Starting experiment 38: random → gaussian_noise'
export EXPERIMENT_NAME='cross_random_to_gaussian_noise'
export TRAIN_CORRUPTION='random'
export EVAL_CORRUPTION='gaussian_noise'

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 --corruption random

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 --corruption random

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption gaussian_noise

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 38: random → gaussian_noise'
sleep 10  # Brief pause between experiments

# Experiment 39: random → motion_blur
echo 'Starting experiment 39: random → motion_blur'
export EXPERIMENT_NAME='cross_random_to_motion_blur'
export TRAIN_CORRUPTION='random'
export EVAL_CORRUPTION='motion_blur'

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 --corruption random

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 --corruption random

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption motion_blur

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 39: random → motion_blur'
sleep 10  # Brief pause between experiments

# Experiment 40: random → defocus_blur
echo 'Starting experiment 40: random → defocus_blur'
export EXPERIMENT_NAME='cross_random_to_defocus_blur'
export TRAIN_CORRUPTION='random'
export EVAL_CORRUPTION='defocus_blur'

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 --corruption random

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 --corruption random

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption defocus_blur

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 40: random → defocus_blur'
sleep 10  # Brief pause between experiments

# Experiment 41: random → uneven_illumination
echo 'Starting experiment 41: random → uneven_illumination'
export EXPERIMENT_NAME='cross_random_to_uneven_illumination'
export TRAIN_CORRUPTION='random'
export EVAL_CORRUPTION='uneven_illumination'

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 --corruption random

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 --corruption random

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption uneven_illumination

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 41: random → uneven_illumination'
sleep 10  # Brief pause between experiments

# Experiment 42: random → smoke_effect
echo 'Starting experiment 42: random → smoke_effect'
export EXPERIMENT_NAME='cross_random_to_smoke_effect'
export TRAIN_CORRUPTION='random'
export EVAL_CORRUPTION='smoke_effect'

# Step 1: Training
python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 --corruption random

# Step 2: Long-short training
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 --corruption random

# Step 3: Generate predictions
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption smoke_effect

# Step 4: Evaluation
cd /home/santhi/Documents/DACAT/src/Cholec80
python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo 'Completed experiment 42: random → smoke_effect'
sleep 10  # Brief pause between experiments
