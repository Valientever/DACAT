# conda activate dacat
# # from ipdb import set_trace
# # gaussian_noise  motion_blur  defocus_blur  uneven_illumination  smoke_effect
# # Create a variabe to store experiment name 

# EXPERIMENT_NAME="11_epoch_w28"
# CORRUPTION_NAME="11_epoch"
# PREDICTION_NAME="predicts"
# set -x
# # Define log file location
# LOG_PATH="/home/santhi/Documents/DACAT/src/Cholec80/results/$EXPERIMENT_NAME"
# LOG_FILE="$LOG_PATH/log_file.txt"
# # LOG_FILE="/home/santhi/Documents/DACAT/src/Cholec80/results/$EXPERIMENT_NAME/log_file.txt"

# # Create log file if it doesn't exist
# mkdir -p "$LOG_PATH"

# touch "$LOG_FILE"
# chmod 666 "$LOG_FILE"

# # Redirect all output (stdout & stderr) to log file and terminal
# exec > >(tee -a "$LOG_FILE") 2>&1

# # Add timestamp at the beginning of the log
# echo "Logging started at $(date)"

# export CUDA_VISIBLE_DEVICES=0

# # cd .../Cholec80/train_scripts
# cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

# # Step 1
# # train/val/test: cuhk 32/8/40; cuhknotest 32/8/0; cuhk4040; 40/0/40
# echo "Starting Step 1....."

# # # gaussian_noise  motion_blur  defocus_blur  uneven_illumination  smoke_effect

# python3 train.py phase --split cuhk4040 --backbone convnextv2 --freeze --workers 28 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME  --step_1 phase_1 --step 1 --epochs 10     #--corruption  #300

# if [ $? -ne 0 ]; then
#     echo "Step 1 failed. Exiting."
#     exit 1
# fi
# echo "Step 1 completed successfully."

# echo "Logged step 1 complete at $(date)"


# echo "Starting Step 2..."


# ## Step 2
# python3 train_longshort.py phase --split cuhk4040 --backbone convnextv2 --workers 28 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME  --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 #--corruption #30 

# if [ $? -ne 0 ]; then
#     echo "Step 2 failed. Exiting."
#     exit 1
# fi
# echo "Step 2 completed successfully."

# echo "Logged step 2 complete at $(date)"


# echo "Starting Step 3....."
# conda activate dacat #pytorch1_13

# export CUDA_VISIBLE_DEVICES=0

# python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 \
#      --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 $PREDICTION_NAME --step 3 # .../checkpoint_best_acc.pth.tar


# if [ $? -ne 0 ]; then
#     echo "Step 3 failed. Exiting."
#     exit 1
# fi
# echo "Step 3 completed successfully."

# echo "Logged step 3 complete at $(date)"


# echo "Starting Step 4....."
# conda activate dacat #pytorch1_13
# cd /home/santhi/Documents/DACAT/src/Cholec80

# export CUDA_VISIBLE_DEVICES=0

# python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name $PREDICTION_NAME


# if [ $? -ne 0 ]; then
#     echo "Step 4 failed. Exiting."
#     exit 1
# fi
# echo "Step 4 completed successfully."
# # # # cp "/home/santhi/Documents/DACAT/src/Cholec80/output/checkpoints/phase/20250112-0858_Step1_cuhk4040Split_lstm_convnextv2_lr0.0001_bs1_seq256_frozen/models/checkpoint_best_acc.pth.tar" "/home/santhi/Documents/DACAT/src/Cholec80/train_scripts/newly_opt_ykx/LongShortNet/long_net_convnextv2.pth.tar"

# echo "Logged step 4 complete at $(date)"
# ###########################################################

# conda activate dacat
# # gaussian_noise  motion_blur  defocus_blur  uneven_illumination  smoke_effect
# # Create a variabe to store experiment name 
# EXPERIMENT_NAME="gaussian_noise_w28"
# CORRUPTION_NAME="gaussian_noise"
# PREDICTION_NAME="predicts"


# # Define log file location
# LOG_PATH="/home/santhi/Documents/DACAT/src/Cholec80/results/$EXPERIMENT_NAME"
# LOG_FILE="$LOG_PATH/log_file.txt"
# # LOG_FILE="/home/santhi/Documents/DACAT/src/Cholec80/results/$EXPERIMENT_NAME/log_file.txt"

# # Create log file if it doesn't exist
# mkdir -p "$LOG_PATH"

# touch "$LOG_FILE"
# chmod 666 "$LOG_FILE"


# # Redirect all output (stdout & stderr) to log file and terminal
# exec > >(tee -a "$LOG_FILE") 2>&1

# # Add timestamp at the beginning of the log
# echo "Logging started at $(date)"

# export CUDA_VISIBLE_DEVICES=0

# # # cd .../Cholec80/train_scripts
# cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

# # # Step 1
# # # train/val/test: cuhk 32/8/40; cuhknotest 32/8/0; cuhk4040; 40/0/40
# echo "Starting Step 1....."

# # # # gaussian_noise  motion_blur  defocus_blur  uneven_illumination  smoke_effect

# python3 train.py phase --split cuhk4040 --backbone convnextv2 --freeze --workers 28 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --corruption $CORRUPTION_NAME --step_1 phase_1 --step 1 --epochs 10     

# if [ $? -ne 0 ]; then
#     echo "Step 1 failed. Exiting."
#     exit 1
# fi
# echo "Step 1 completed successfully."

# echo "Logged step 1 complete at $(date)"


# echo "Starting Step 2..."


# ## Step 2
# python3 train_longshort.py phase --split cuhk4040 --backbone convnextv2 --workers 28 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --corruption $CORRUPTION_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 #--corruption #30 

# if [ $? -ne 0 ]; then
#     echo "Step 2 failed. Exiting."
#     exit 1
# fi
# echo "Step 2 completed successfully."

# echo "Logged step 2 complete at $(date)"


# echo "Starting Step 3....."
# conda activate dacat #pytorch1_13

# export CUDA_VISIBLE_DEVICES=0

# python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 \
#      --resume 1 --experiment_name $EXPERIMENT_NAME --corruption $CORRUPTION_NAME --step_1 phase_2 --step_3 $PREDICTION_NAME --step 3 # .../checkpoint_best_acc.pth.tar


# if [ $? -ne 0 ]; then
#     echo "Step 3 failed. Exiting."
#     exit 1
# fi
# echo "Step 3 completed successfully."

# echo "Logged step 3 complete at $(date)"


# echo "Starting Step 4....."
# conda activate dacat #pytorch1_13
# cd /home/santhi/Documents/DACAT/src/Cholec80

# export CUDA_VISIBLE_DEVICES=0

# python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name $PREDICTION_NAME


# if [ $? -ne 0 ]; then
#     echo "Step 4 failed. Exiting."
#     exit 1
# fi
# echo "Step 4 completed successfully."
# # # # cp "/home/santhi/Documents/DACAT/src/Cholec80/output/checkpoints/phase/20250112-0858_Step1_cuhk4040Split_lstm_convnextv2_lr0.0001_bs1_seq256_frozen/models/checkpoint_best_acc.pth.tar" "/home/santhi/Documents/DACAT/src/Cholec80/train_scripts/newly_opt_ykx/LongShortNet/long_net_convnextv2.pth.tar"

# echo "Logged step 4 complete at $(date)"

# #######################################################################

# conda activate dacat
# # gaussian_noise  motion_blur  defocus_blur  uneven_illumination  smoke_effect
# # Create a variabe to store experiment name 
# EXPERIMENT_NAME="motion_blur_w28"
# CORRUPTION_NAME="motion_blur"
# PREDICTION_NAME="predicts"


# # Define log file location
# LOG_PATH="/home/santhi/Documents/DACAT/src/Cholec80/results/$EXPERIMENT_NAME"
# LOG_FILE="$LOG_PATH/log_file.txt"
# # LOG_FILE="/home/santhi/Documents/DACAT/src/Cholec80/results/$EXPERIMENT_NAME/log_file.txt"

# # Create log file if it doesn't exist
# mkdir -p "$LOG_PATH"

# touch "$LOG_FILE"
# chmod 666 "$LOG_FILE"


# # Redirect all output (stdout & stderr) to log file and terminal
# exec > >(tee -a "$LOG_FILE") 2>&1

# # Add timestamp at the beginning of the log
# echo "Logging started at $(date)"

# export CUDA_VISIBLE_DEVICES=0

# # # cd .../Cholec80/train_scripts
# cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

# # Step 1
# # train/val/test: cuhk 32/8/40; cuhknotest 32/8/0; cuhk4040; 40/0/40
# echo "Starting Step 1....."

# # # gaussian_noise  motion_blur  defocus_blur  uneven_illumination  smoke_effect

# python3 train.py phase --split cuhk4040 --backbone convnextv2 --freeze --workers 28 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --corruption $CORRUPTION_NAME --step_1 phase_1 --step 1 --epochs 10     #--corruption  #300

# if [ $? -ne 0 ]; then
#     echo "Step 1 failed. Exiting."
#     exit 1
# fi
# echo "Step 1 completed successfully."

# echo "Logged step 1 complete at $(date)"


# echo "Starting Step 2..."


# ## Step 2
# python3 train_longshort.py phase --split cuhk4040 --backbone convnextv2 --workers 28 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --corruption $CORRUPTION_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 #--corruption #30 

# if [ $? -ne 0 ]; then
#     echo "Step 2 failed. Exiting."
#     exit 1
# fi
# echo "Step 2 completed successfully."

# echo "Logged step 2 complete at $(date)"


# echo "Starting Step 3....."
# conda activate dacat #pytorch1_13

# export CUDA_VISIBLE_DEVICES=0

# python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 \
#      --resume 1 --experiment_name $EXPERIMENT_NAME --corruption $CORRUPTION_NAME --step_1 phase_2 --step_3 $PREDICTION_NAME --step 3 # .../checkpoint_best_acc.pth.tar


# if [ $? -ne 0 ]; then
#     echo "Step 3 failed. Exiting."
#     exit 1
# fi
# echo "Step 3 completed successfully."

# echo "Logged step 3 complete at $(date)"


# echo "Starting Step 4....."
# conda activate dacat #pytorch1_13
# cd /home/santhi/Documents/DACAT/src/Cholec80

# export CUDA_VISIBLE_DEVICES=0

# python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name $PREDICTION_NAME


# if [ $? -ne 0 ]; then
#     echo "Step 4 failed. Exiting."
#     exit 1
# fi
# echo "Step 4 completed successfully."
# # # # cp "/home/santhi/Documents/DACAT/src/Cholec80/output/checkpoints/phase/20250112-0858_Step1_cuhk4040Split_lstm_convnextv2_lr0.0001_bs1_seq256_frozen/models/checkpoint_best_acc.pth.tar" "/home/santhi/Documents/DACAT/src/Cholec80/train_scripts/newly_opt_ykx/LongShortNet/long_net_convnextv2.pth.tar"

# echo "Logged step 4 complete at $(date)"

# #######################################################################

# conda activate dacat
# # gaussian_noise  motion_blur  defocus_blur  uneven_illumination  smoke_effect
# # Create a variabe to store experiment name 
# EXPERIMENT_NAME="defocus_blur_w28"
# CORRUPTION_NAME="defocus_blur"
# PREDICTION_NAME="predicts"


# # Define log file location
# LOG_PATH="/home/santhi/Documents/DACAT/src/Cholec80/results/$EXPERIMENT_NAME"
# LOG_FILE="$LOG_PATH/log_file.txt"
# # LOG_FILE="/home/santhi/Documents/DACAT/src/Cholec80/results/$EXPERIMENT_NAME/log_file.txt"

# # Create log file if it doesn't exist
# mkdir -p "$LOG_PATH"

# touch "$LOG_FILE"
# chmod 666 "$LOG_FILE"


# # Redirect all output (stdout & stderr) to log file and terminal
# exec > >(tee -a "$LOG_FILE") 2>&1

# # Add timestamp at the beginning of the log
# echo "Logging started at $(date)"

# export CUDA_VISIBLE_DEVICES=0

# # # cd .../Cholec80/train_scripts
# cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

# # Step 1
# # train/val/test: cuhk 32/8/40; cuhknotest 32/8/0; cuhk4040; 40/0/40
# echo "Starting Step 1....."

# # # gaussian_noise  motion_blur  defocus_blur  uneven_illumination  smoke_effect

# python3 train.py phase --split cuhk4040 --backbone convnextv2 --freeze --workers 28 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --corruption $CORRUPTION_NAME --step_1 phase_1 --step 1 --epochs 10     #--corruption  #300

# if [ $? -ne 0 ]; then
#     echo "Step 1 failed. Exiting."
#     exit 1
# fi
# echo "Step 1 completed successfully."

# echo "Logged step 1 complete at $(date)"


# echo "Starting Step 2..."


# ## Step 2
# python3 train_longshort.py phase --split cuhk4040 --backbone convnextv2 --workers 28 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --corruption $CORRUPTION_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 #--corruption #30 

# if [ $? -ne 0 ]; then
#     echo "Step 2 failed. Exiting."
#     exit 1
# fi
# echo "Step 2 completed successfully."

# echo "Logged step 2 complete at $(date)"


# echo "Starting Step 3....."
# conda activate dacat #pytorch1_13

# export CUDA_VISIBLE_DEVICES=0

# python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 \
#      --resume 1 --experiment_name $EXPERIMENT_NAME --corruption $CORRUPTION_NAME --step_1 phase_2 --step_3 $PREDICTION_NAME --step 3 # .../checkpoint_best_acc.pth.tar


# if [ $? -ne 0 ]; then
#     echo "Step 3 failed. Exiting."
#     exit 1
# fi
# echo "Step 3 completed successfully."

# echo "Logged step 3 complete at $(date)"


# echo "Starting Step 4....."
# conda activate dacat #pytorch1_13
# cd /home/santhi/Documents/DACAT/src/Cholec80

# export CUDA_VISIBLE_DEVICES=0

# python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name $PREDICTION_NAME


# if [ $? -ne 0 ]; then
#     echo "Step 4 failed. Exiting."
#     exit 1
# fi
# echo "Step 4 completed successfully."
# # # # cp "/home/santhi/Documents/DACAT/src/Cholec80/output/checkpoints/phase/20250112-0858_Step1_cuhk4040Split_lstm_convnextv2_lr0.0001_bs1_seq256_frozen/models/checkpoint_best_acc.pth.tar" "/home/santhi/Documents/DACAT/src/Cholec80/train_scripts/newly_opt_ykx/LongShortNet/long_net_convnextv2.pth.tar"

# echo "Logged step 4 complete at $(date)"

# #######################################################################

# conda activate dacat
# # gaussian_noise  motion_blur  defocus_blur  uneven_illumination  smoke_effect
# # Create a variabe to store experiment name 
# EXPERIMENT_NAME="uneven_illumination_w28"
# CORRUPTION_NAME="uneven_illumination"
# PREDICTION_NAME="predicts"
# PREDICTION_NAME="predicts"


# # Define log file location
# LOG_PATH="/home/santhi/Documents/DACAT/src/Cholec80/results/$EXPERIMENT_NAME"
# LOG_FILE="$LOG_PATH/log_file.txt"
# # LOG_FILE="/home/santhi/Documents/DACAT/src/Cholec80/results/$EXPERIMENT_NAME/log_file.txt"

# # Create log file if it doesn't exist
# mkdir -p "$LOG_PATH"

# touch "$LOG_FILE"
# chmod 666 "$LOG_FILE"

# # Redirect all output (stdout & stderr) to log file and terminal
# exec > >(tee -a "$LOG_FILE") 2>&1

# # Add timestamp at the beginning of the log
# echo "Logging started at $(date)"

# export CUDA_VISIBLE_DEVICES=0

# # # cd .../Cholec80/train_scripts
# cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

# # Step 1
# # train/val/test: cuhk 32/8/40; cuhknotest 32/8/0; cuhk4040; 40/0/40
# echo "Starting Step 1....."

# # # gaussian_noise  motion_blur  defocus_blur  uneven_illumination  smoke_effect

# python3 train.py phase --split cuhk4040 --backbone convnextv2 --freeze --workers 28 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --corruption $CORRUPTION_NAME --step_1 phase_1 --step 1 --epochs 10     #--corruption  #300

# if [ $? -ne 0 ]; then
#     echo "Step 1 failed. Exiting."
#     exit 1
# fi
# echo "Step 1 completed successfully."

# echo "Logged step 1 complete at $(date)"


# echo "Starting Step 2..."


# ## Step 2
# python3 train_longshort.py phase --split cuhk4040 --backbone convnextv2 --workers 28 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --corruption $CORRUPTION_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 #--corruption #30 

# if [ $? -ne 0 ]; then
#     echo "Step 2 failed. Exiting."
#     exit 1
# fi
# echo "Step 2 completed successfully."

# echo "Logged step 2 complete at $(date)"


# echo "Starting Step 3....."
# conda activate dacat #pytorch1_13

# export CUDA_VISIBLE_DEVICES=0

# python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 \
#      --resume 1 --experiment_name $EXPERIMENT_NAME --corruption $CORRUPTION_NAME --step_1 phase_2 --step_3 $PREDICTION_NAME --step 3 # .../checkpoint_best_acc.pth.tar


# if [ $? -ne 0 ]; then
#     echo "Step 3 failed. Exiting."
#     exit 1
# fi
# echo "Step 3 completed successfully."

# echo "Logged step 3 complete at $(date)"


# echo "Starting Step 4....."
# conda activate dacat #pytorch1_13
# cd /home/santhi/Documents/DACAT/src/Cholec80

# export CUDA_VISIBLE_DEVICES=0

# python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name $PREDICTION_NAME


# if [ $? -ne 0 ]; then
#     echo "Step 4 failed. Exiting."
#     exit 1
# fi
# echo "Step 4 completed successfully."
# # # # cp "/home/santhi/Documents/DACAT/src/Cholec80/output/checkpoints/phase/20250112-0858_Step1_cuhk4040Split_lstm_convnextv2_lr0.0001_bs1_seq256_frozen/models/checkpoint_best_acc.pth.tar" "/home/santhi/Documents/DACAT/src/Cholec80/train_scripts/newly_opt_ykx/LongShortNet/long_net_convnextv2.pth.tar"

# echo "Logged step 4 complete at $(date)"

# #######################################################################

# conda activate dacat
# # gaussian_noise  motion_blur  defocus_blur  uneven_illumination  smoke_effect
# # Create a variabe to store experiment name 
# EXPERIMENT_NAME="smoke_effect_w28"
# CORRUPTION_NAME="smoke_effect"
# PREDICTION_NAME="predicts"


# # Define log file location
# LOG_PATH="/home/santhi/Documents/DACAT/src/Cholec80/results/$EXPERIMENT_NAME"
# LOG_FILE="$LOG_PATH/log_file.txt"
# # LOG_FILE="/home/santhi/Documents/DACAT/src/Cholec80/results/$EXPERIMENT_NAME/log_file.txt"

# # Create log file if it doesn't exist
# mkdir -p "$LOG_PATH"

# touch "$LOG_FILE"
# chmod 666 "$LOG_FILE"


# # Redirect all output (stdout & stderr) to log file and terminal
# exec > >(tee -a "$LOG_FILE") 2>&1

# # Add timestamp at the beginning of the log
# echo "Logging started at $(date)"

# export CUDA_VISIBLE_DEVICES=0

# # # cd .../Cholec80/train_scripts
# cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

# # Step 1
# # train/val/test: cuhk 32/8/40; cuhknotest 32/8/0; cuhk4040; 40/0/40
# echo "Starting Step 1....."

# # # gaussian_noise  motion_blur  defocus_blur  uneven_illumination  smoke_effect

# python3 train.py phase --split cuhk4040 --backbone convnextv2 --freeze --workers 28 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --corruption $CORRUPTION_NAME --step_1 phase_1 --step 1 --epochs 10     #--corruption  #300

# if [ $? -ne 0 ]; then
#     echo "Step 1 failed. Exiting."
#     exit 1
# fi
# echo "Step 1 completed successfully."

# echo "Logged step 1 complete at $(date)"


# echo "Starting Step 2..."


# ## Step 2
# python3 train_longshort.py phase --split cuhk4040 --backbone convnextv2 --workers 28 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --corruption $CORRUPTION_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 #--corruption #30 

# if [ $? -ne 0 ]; then
#     echo "Step 2 failed. Exiting."
#     exit 1
# fi
# echo "Step 2 completed successfully."

# echo "Logged step 2 complete at $(date)"


# echo "Starting Step 3....."
# conda activate dacat #pytorch1_13

# export CUDA_VISIBLE_DEVICES=0

# python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 \
#      --resume 1 --experiment_name $EXPERIMENT_NAME --corruption $CORRUPTION_NAME --step_1 phase_2 --step_3 $PREDICTION_NAME --step 3 # .../checkpoint_best_acc.pth.tar


# if [ $? -ne 0 ]; then
#     echo "Step 3 failed. Exiting."
#     exit 1
# fi
# echo "Step 3 completed successfully."

# echo "Logged step 3 complete at $(date)"


# echo "Starting Step 4....."
# conda activate dacat #pytorch1_13
# cd /home/santhi/Documents/DACAT/src/Cholec80

# export CUDA_VISIBLE_DEVICES=0

# python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name $PREDICTION_NAME


# if [ $? -ne 0 ]; then
#     echo "Step 4 failed. Exiting."
#     exit 1
# fi
# echo "Step 4 completed successfully."
# # # # cp "/home/santhi/Documents/DACAT/src/Cholec80/output/checkpoints/phase/20250112-0858_Step1_cuhk4040Split_lstm_convnextv2_lr0.0001_bs1_seq256_frozen/models/checkpoint_best_acc.pth.tar" "/home/santhi/Documents/DACAT/src/Cholec80/train_scripts/newly_opt_ykx/LongShortNet/long_net_convnextv2.pth.tar"

# echo "Logged step 4 complete at $(date)"

# #######################################################################

conda activate dacat
# gaussian_noise  motion_blur  defocus_blur  uneven_illumination  smoke_effect  random
# Create a variabe to store experiment name 
EXPERIMENT_NAME="1_epoch_w28"
# CORRUPTION_NAME="random"
PREDICTION_NAME="predicts"


# Define log file location
LOG_PATH="/home/santhi/Documents/DACAT/src/Cholec80/results/$EXPERIMENT_NAME"
LOG_FILE="$LOG_PATH/log_file.txt"
# LOG_FILE="/home/santhi/Documents/DACAT/src/Cholec80/results/$EXPERIMENT_NAME/log_file.txt"

# Create log file if it doesn't exist
mkdir -p "$LOG_PATH"

touch "$LOG_FILE"
chmod 666 "$LOG_FILE"


# Redirect all output (stdout & stderr) to log file and terminal
exec > >(tee -a "$LOG_FILE") 2>&1

# Add timestamp at the beginning of the log
echo "Logging started at $(date)"

export CUDA_VISIBLE_DEVICES=0

# # cd .../Cholec80/train_scripts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

# Step 1
# train/val/test: cuhk 32/8/40; cuhknotest 32/8/0; cuhk4040; 40/0/40
echo "Starting Step 1....."

# # gaussian_noise  motion_blur  defocus_blur  uneven_illumination  smoke_effect

python3 train.py phase --split cuhk4040 --backbone convnextv2 --freeze --workers 28 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 1    #--corruption  #300

if [ $? -ne 0 ]; then
    echo "Step 1 failed. Exiting."
    exit 1
fi
echo "Step 1 completed successfully."

echo "Logged step 1 complete at $(date)"


echo "Starting Step 2..."


## Step 2
python3 train_longshort.py phase --split cuhk4040 --backbone convnextv2 --workers 28 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 1 #--corruption #30 

if [ $? -ne 0 ]; then
    echo "Step 2 failed. Exiting."
    exit 1
fi
echo "Step 2 completed successfully."

echo "Logged step 2 complete at $(date)"


echo "Starting Step 3....."
conda activate dacat #pytorch1_13

export CUDA_VISIBLE_DEVICES=0

python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 \
     --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 $PREDICTION_NAME --step 3 # .../checkpoint_best_acc.pth.tar


if [ $? -ne 0 ]; then
    echo "Step 3 failed. Exiting."
    exit 1
fi
echo "Step 3 completed successfully."

echo "Logged step 3 complete at $(date)"


echo "Starting Step 4....."
conda activate dacat #pytorch1_13
cd /home/santhi/Documents/DACAT/src/Cholec80

export CUDA_VISIBLE_DEVICES=0

python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name $PREDICTION_NAME


if [ $? -ne 0 ]; then
    echo "Step 4 failed. Exiting."
    exit 1
fi
echo "Step 4 completed successfully."
# # # cp "/home/santhi/Documents/DACAT/src/Cholec80/output/checkpoints/phase/20250112-0858_Step1_cuhk4040Split_lstm_convnextv2_lr0.0001_bs1_seq256_frozen/models/checkpoint_best_acc.pth.tar" "/home/santhi/Documents/DACAT/src/Cholec80/train_scripts/newly_opt_ykx/LongShortNet/long_net_convnextv2.pth.tar"

echo "Logged step 4 complete at $(date)"
