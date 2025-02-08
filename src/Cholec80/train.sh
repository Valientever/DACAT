# conda activate pytorch
conda activate dacat
python - <<EOF
import torch
# print("Torch version:", torch.__version__)
# print("CUDA available:", torch.cuda.is_available())
# print("Number of GPUs:", torch.cuda.device_count())
# print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")
# if torch.cuda.is_available():
#     print("Using Device:", torch.cuda.get_device_name(0))
device = torch.device('cpu')
print(f"Using device: {device}")
EOF



# export CUDA_VISIBLE_DEVICES=0

# # cd .../Cholec80/train_scripts
cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts

## Step 1
# train/val/test: cuhk 32/8/40; cuhknotest 32/8/0; cuhk4040; 40/0/40
echo "Starting Step 1....."



python3 train.py phase --split cuhk4040 --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1  --experiment_name train_original_1  --epochs 1 #300

if [ $? -ne 0 ]; then
    echo "Step 1 failed. Exiting."
    exit 1
fi
echo "Step 1 completed successfully."

echo "Starting Step 2..."

# # cp "/home/santhi/Documents/DACAT/src/Cholec80/output/checkpoints/phase/20250112-0858_Step1_cuhk4040Split_lstm_convnextv2_lr0.0001_bs1_seq256_frozen/models/checkpoint_best_acc.pth.tar" "/home/santhi/Documents/DACAT/src/Cholec80/train_scripts/newly_opt_ykx/LongShortNet/long_net_convnextv2.pth.tar"

# # cp "/home/santhi/Documents/DACAT/src/Cholec80/results/test_exp/20250207-0712_Step1_cuhk4040Split_lstm_convnextv2_lr0.0001_bs1_seq256_frozen/models/checkpoint_best_acc.pth.tar" "/home/santhi/Documents/DACAT/src/Cholec80/train_scripts/newly_opt_ykx/LongShortNet/long_net_convnextv2.pth.tar"

## Step 2
## You need to use trained model in Step 1, and saved in .../train_scripts/newly_opt_ykx/LongShortNet/long_net_convnextv2.pth.tar
python3 train_longshort.py phase --split cuhk4040 --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name train_original_1 --step_1 phase --epochs 1 #30 

if [ $? -ne 0 ]; then
    echo "Step 2 failed. Exiting."
    exit 1
fi
echo "Step 2 completed successfully."