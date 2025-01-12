conda activate dacat #pytorch1_13

export CUDA_VISIBLE_DEVICES=0

cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts #.../Cholec80/train_scripts


python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 \
    --resume /home/santhi/Documents/DACAT/checkpoints/Cholec80/checkpoint_best_acc.pth.tar # .../checkpoint_best_acc.pth.tar
