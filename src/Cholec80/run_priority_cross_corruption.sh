#!/bin/bash
# Priority cross-corruption experiments

# Priority Experiment 1: clean → gaussian_noise
echo 'Starting priority experiment 1: clean → gaussian_noise'
export EXPERIMENT_NAME='priority_clean_to_gaussian_noise'

python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 5 
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 5 
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption gaussian_noise
cd /home/santhi/Documents/DACAT/src/Cholec80 && python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts && cd train_scripts

# Priority Experiment 2: clean → motion_blur
echo 'Starting priority experiment 2: clean → motion_blur'
export EXPERIMENT_NAME='priority_clean_to_motion_blur'

python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 5 
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 5 
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption motion_blur
cd /home/santhi/Documents/DACAT/src/Cholec80 && python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts && cd train_scripts

# Priority Experiment 3: clean → defocus_blur
echo 'Starting priority experiment 3: clean → defocus_blur'
export EXPERIMENT_NAME='priority_clean_to_defocus_blur'

python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 5 
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 5 
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption defocus_blur
cd /home/santhi/Documents/DACAT/src/Cholec80 && python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts && cd train_scripts

# Priority Experiment 4: clean → smoke_effect
echo 'Starting priority experiment 4: clean → smoke_effect'
export EXPERIMENT_NAME='priority_clean_to_smoke_effect'

python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 5 
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 5 
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption smoke_effect
cd /home/santhi/Documents/DACAT/src/Cholec80 && python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts && cd train_scripts

# Priority Experiment 5: gaussian_noise → motion_blur
echo 'Starting priority experiment 5: gaussian_noise → motion_blur'
export EXPERIMENT_NAME='priority_gaussian_noise_to_motion_blur'

python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 5 --corruption gaussian_noise
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 5 --corruption gaussian_noise
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption motion_blur
cd /home/santhi/Documents/DACAT/src/Cholec80 && python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts && cd train_scripts

# Priority Experiment 6: motion_blur → defocus_blur
echo 'Starting priority experiment 6: motion_blur → defocus_blur'
export EXPERIMENT_NAME='priority_motion_blur_to_defocus_blur'

python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 5 --corruption motion_blur
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 5 --corruption motion_blur
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption defocus_blur
cd /home/santhi/Documents/DACAT/src/Cholec80 && python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts && cd train_scripts

# Priority Experiment 7: defocus_blur → uneven_illumination
echo 'Starting priority experiment 7: defocus_blur → uneven_illumination'
export EXPERIMENT_NAME='priority_defocus_blur_to_uneven_illumination'

python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 5 --corruption defocus_blur
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 5 --corruption defocus_blur
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 --corruption uneven_illumination
cd /home/santhi/Documents/DACAT/src/Cholec80 && python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts && cd train_scripts

# Priority Experiment 8: gaussian_noise → clean
echo 'Starting priority experiment 8: gaussian_noise → clean'
export EXPERIMENT_NAME='priority_gaussian_noise_to_clean'

python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 5 --corruption gaussian_noise
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 5 --corruption gaussian_noise
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 
cd /home/santhi/Documents/DACAT/src/Cholec80 && python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts && cd train_scripts

# Priority Experiment 9: motion_blur → clean
echo 'Starting priority experiment 9: motion_blur → clean'
export EXPERIMENT_NAME='priority_motion_blur_to_clean'

python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 5 --corruption motion_blur
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 5 --corruption motion_blur
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 
cd /home/santhi/Documents/DACAT/src/Cholec80 && python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts && cd train_scripts

# Priority Experiment 10: defocus_blur → clean
echo 'Starting priority experiment 10: defocus_blur → clean'
export EXPERIMENT_NAME='priority_defocus_blur_to_clean'

python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 5 --corruption defocus_blur
python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 5 --corruption defocus_blur
python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 
cd /home/santhi/Documents/DACAT/src/Cholec80 && python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts && cd train_scripts
