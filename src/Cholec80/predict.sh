conda activate dacat #pytorch1_13

export CUDA_VISIBLE_DEVICES=0

cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts #.../Cholec80/train_scripts

# add another input here, so when you give the experiment-name, it will get the best model from that experiment
# python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 \
#     --resume /home/santhi/Documents/DACAT/checkpoints/Cholec80/checkpoint_best_acc.pth.tar  # .../checkpoint_best_acc.pth.tar

python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 \
     --resume /home/santhi/Documents/DACAT/src/Cholec80/results/data_path/data/20250210-0946_DACAT_cuhk4040Split_lstm_convnextv2_lr1e-05_bs1_seq64_e2e/models/checkpoint_best_acc.pth.tar --experiment_name data_path # .../checkpoint_best_acc.pth.tar
