conda activate dacat

# gaussian_noise  motion_blur  defocus_blur  uneven_illumination  smoke_effect

EXPERIMENT_NAME="random_w28"
PREDICTION_NAME="predict_base"
set -x
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

# cd .../Cholec80/train_scripts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo "Starting Step 3....."
conda activate dacat #pytorch1_13

export CUDA_VISIBLE_DEVICES=0

python3 save_predictions_onlinev2_longshort.py  phase --split cuhk --backbone convnextv2 --seq_len 1 \
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

echo "Logged step 4 complete at $(date)"


######################################################
# conda activate dacat

# gaussian_noise  motion_blur  defocus_blur  uneven_illumination  smoke_effect

EXPERIMENT_NAME="random_w28"
CORRUPTION_NAME="gaussian_noise"
PREDICTION_NAME="gn_10_w28_predicts"
set -x
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

# cd .../Cholec80/train_scripts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo "Starting Step 3....."
conda activate dacat #pytorch1_13

export CUDA_VISIBLE_DEVICES=0

python3 save_predictions_onlinev2_longshort.py  phase --split cuhk --backbone convnextv2 --seq_len 1 \
     --resume 1 --experiment_name $EXPERIMENT_NAME --corruption $CORRUPTION_NAME --step_1 phase_2 --step_3 $PREDICTION_NAME --step 3 # .../checkpoint_best_acc.pth.tar


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

echo "Logged step 4 complete at $(date)"


######################################################conda activate dacat

# gaussian_noise  motion_blur  defocus_blur  uneven_illumination  smoke_effect

EXPERIMENT_NAME="random_w28"
CORRUPTION_NAME="motion_blur"
PREDICTION_NAME="mb_10_w28_predicts"
set -x
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

# cd .../Cholec80/train_scripts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo "Starting Step 3....."
conda activate dacat #pytorch1_13

export CUDA_VISIBLE_DEVICES=0

python3 save_predictions_onlinev2_longshort.py  phase --split cuhk --backbone convnextv2 --seq_len 1 \
     --resume 1 --experiment_name $EXPERIMENT_NAME --corruption $CORRUPTION_NAME --step_1 phase_2 --step_3 $PREDICTION_NAME --step 3 # .../checkpoint_best_acc.pth.tar


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

echo "Logged step 4 complete at $(date)"


######################################################conda activate dacat

# gaussian_noise  motion_blur  defocus_blur  uneven_illumination  smoke_effect

EXPERIMENT_NAME="random_w28"
CORRUPTION_NAME="defocus_blur"
PREDICTION_NAME="db_10_w28_predicts"
set -x
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

# cd .../Cholec80/train_scripts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo "Starting Step 3....."
conda activate dacat #pytorch1_13

export CUDA_VISIBLE_DEVICES=0

python3 save_predictions_onlinev2_longshort.py  phase --split cuhk --backbone convnextv2 --seq_len 1 \
     --resume 1 --experiment_name $EXPERIMENT_NAME --corruption $CORRUPTION_NAME --step_1 phase_2 --step_3 $PREDICTION_NAME --step 3 # .../checkpoint_best_acc.pth.tar


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

echo "Logged step 4 complete at $(date)"


######################################################conda activate dacat

# gaussian_noise  motion_blur  defocus_blur  uneven_illumination  smoke_effect

EXPERIMENT_NAME="random_w28"
CORRUPTION_NAME="uneven_illumination"
PREDICTION_NAME="ui_10_w28_predicts"
set -x
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

# cd .../Cholec80/train_scripts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo "Starting Step 3....."
conda activate dacat #pytorch1_13

export CUDA_VISIBLE_DEVICES=0

python3 save_predictions_onlinev2_longshort.py  phase --split cuhk --backbone convnextv2 --seq_len 1 \
     --resume 1 --experiment_name $EXPERIMENT_NAME --corruption $CORRUPTION_NAME --step_1 phase_2 --step_3 $PREDICTION_NAME --step 3 # .../checkpoint_best_acc.pth.tar


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

echo "Logged step 4 complete at $(date)"


######################################################conda activate dacat

# gaussian_noise  motion_blur  defocus_blur  uneven_illumination  smoke_effect

EXPERIMENT_NAME="random_w28"
CORRUPTION_NAME="smoke_effect"
PREDICTION_NAME="se_10_w28_predicts"
set -x
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

# cd .../Cholec80/train_scripts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo "Starting Step 3....."
conda activate dacat #pytorch1_13

export CUDA_VISIBLE_DEVICES=0

python3 save_predictions_onlinev2_longshort.py  phase --split cuhk --backbone convnextv2 --seq_len 1 \
     --resume 1 --experiment_name $EXPERIMENT_NAME --corruption $CORRUPTION_NAME --step_1 phase_2 --step_3 $PREDICTION_NAME --step 3 # .../checkpoint_best_acc.pth.tar


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

echo "Logged step 4 complete at $(date)"


# ######################################################conda activate dacat

# # gaussian_noise  motion_blur  defocus_blur  uneven_illumination  smoke_effect

EXPERIMENT_NAME="random_w28"
CORRUPTION_NAME="random"
PREDICTION_NAME="random_10_w28_predicts"
set -x
LOG_PATH="/home/santhi/Documents/DACAT/src/Cholec80/results/$EXPERIMENT_NAME"
LOG_FILE="$LOG_PATH/log_file.txt"
# # LOG_FILE="/home/santhi/Documents/DACAT/src/Cholec80/results/$EXPERIMENT_NAME/log_file.txt"

# # Create log file if it doesn't exist
# mkdir -p "$LOG_PATH"

touch "$LOG_FILE"
chmod 666 "$LOG_FILE"

# # Redirect all output (stdout & stderr) to log file and terminal
exec > >(tee -a "$LOG_FILE") 2>&1

# # Add timestamp at the beginning of the log
echo "Logging started at $(date)"

export CUDA_VISIBLE_DEVICES=0

# # cd .../Cholec80/train_scripts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

echo "Starting Step 3....."
conda activate dacat #pytorch1_13

# export CUDA_VISIBLE_DEVICES=0

python3 save_predictions_onlinev2_longshort.py  phase --split cuhk --backbone convnextv2 --seq_len 1 \
     --resume 1 --experiment_name $EXPERIMENT_NAME --corruption $CORRUPTION_NAME --step_1 phase_2 --step_3 $PREDICTION_NAME --step 3 # .../checkpoint_best_acc.pth.tar


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

echo "Logged step 4 complete at $(date)"


######################################################