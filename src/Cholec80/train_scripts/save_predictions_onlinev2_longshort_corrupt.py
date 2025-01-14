import torch
from random import shuffle
from tqdm import tqdm
from options_train import parser
from dataloader import prepare_dataset, prepare_image_features, prepare_batch
from newly_opt_ykx.dataloader import Cholec80Test
from model_anticipation import AnticipationModel
from newly_opt_ykx.LongShortNet.model_phase_maxr_v1_maxca import PhaseModel
import util_train as util
import os
import pandas as pd
import cv2
import numpy as np

opts = parser.parse_args()

# assumes <opts.resume> has form "output/checkpoints/<task>/<trial_name>/models/<checkpoint>.pth.tar"
suffix = 'predv2_DACAT'

out_folder = os.path.dirname(os.path.dirname(opts.resume)).replace('/checkpoints', '/predictions_corrupt/')
print(f'opts.resume: {opts.resume} \n out_folder: {out_folder}')

gt_folder = os.path.join(out_folder, 'gt_corrupt')
print(f'gt_folder: {gt_folder}')
os.makedirs(gt_folder, exist_ok=True)

# Base prediction folder
pred_folder_base = os.path.join(out_folder, suffix)
print(f'Base pred_folder: {pred_folder_base}')
os.makedirs(pred_folder_base, exist_ok=True)

if opts.task == 'anticipation':
    model = AnticipationModel(opts, train=False)
if opts.task == 'phase':
    model = PhaseModel(opts, train=False)

if opts.only_temporal:
    _, _, test_set = prepare_image_features(model.net_short, opts, test_mode=True)
else:
    data_folder = '../data/frames_1fps/'
    op_paths = [os.path.join(data_folder, op) for op in os.listdir(data_folder)]
    if opts.split == 'cuhk':
        op_paths.sort(key=os.path.basename)
        test_set = []
        for op_path in op_paths[40:42]:  # Adjusted for smaller range during debugging
            ID = os.path.basename(op_path)
            if os.path.isdir(op_path):
                test_set.append((ID, op_path))


def apply_corruption(frame, corrupt_mode):
    """
    Applies specified corruption (0..6) to a single frame (NumPy array).
    """
    if corrupt_mode == 0:
        return frame  # No corruption

    corrupted = frame.copy()

    if corrupt_mode == 1 or corrupt_mode == 6:
        alpha = 0.3
        smoke_color = (200, 200, 200)  # BGR
        overlay = np.full_like(corrupted, smoke_color, dtype=np.uint8)
        print("Applying smoke effect")
        corrupted = cv2.addWeighted(overlay, alpha, corrupted, 1 - alpha, 0)

    if corrupt_mode == 2 or corrupt_mode == 6:
        print("Applying defocus blur")
        corrupted = cv2.GaussianBlur(corrupted, (9, 9), 5)

    if corrupt_mode == 3 or corrupt_mode == 6:
        size = 25 #15
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
        kernel_motion_blur /= size
        print("Applying motion blur")
        corrupted = cv2.filter2D(corrupted, -1, kernel_motion_blur)

    if corrupt_mode == 4 or corrupt_mode == 6:
        rows, cols, _ = corrupted.shape
        gradient = np.linspace(1.0, 0.6, cols, dtype=np.float32)
        mask = np.tile(gradient, (rows, 1))
        print("Applying uneven illumination")
        for c in range(3):
            corrupted[:, :, c] = corrupted[:, :, c].astype(np.float32) * mask
        corrupted = np.clip(corrupted, 0, 255).astype(np.uint8)

    if corrupt_mode == 5 or corrupt_mode == 6:
        row, col, ch = corrupted.shape
        mean = 0
        sigma = 10
        print("Applying Gaussian noise")
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        corrupted = corrupted.astype(np.float32)
        corrupted += gauss
        corrupted = np.clip(corrupted, 0, 255).astype(np.uint8)

    return corrupted


def Unitconversion(flops, params, throughout):
    print("params : {} M".format(round(params / (1000**2), 2)))
    print("flop : {} G".format(round(flops / (1000**3), 2)))
    print("throughout: {} Images/Min".format(throughout * 60))


def weight_test(model, x):
    import time
    start_time = time.time()
    _ = model(x)
    end_time = time.time()
    need_time = end_time - start_time
    from thop import profile

    flops, params = profile(model, inputs=(x,))
    throughout = round(x.shape[0] / (need_time / 1), 3)
    return flops, params, throughout


with torch.no_grad():
    if opts.cheat:
        model.net_short.train()
    else:
        model.net_short.eval()
        model.net_long.eval()

    # Always include corruption mode 0 (no corruption)
    corruption_modes = [0]
    if opts.corrupt > 0:
        corruption_modes.append(opts.corrupt)

    for corrupt_mode in corruption_modes:
        corr_subfolder = f"{suffix}_corrupt_{corrupt_mode}"
        pred_folder = os.path.join(out_folder, corr_subfolder)
        os.makedirs(pred_folder, exist_ok=True)
        print(f"\n=== Generating predictions with corruption={corrupt_mode}, storing in {pred_folder} ===\n")

        for ID, op_path in test_set:
            predictions = []
            labels = []

            if not opts.image_based:
                model.net_short.temporal_head.reset()
                model.net_short.CA_temp_head.reset()
            model.net_long.cache_reset()
            model.net_short.cache_reset()

            model.metric_meter['test'].start_new_op()
            offline_cholec80_test = Cholec80Test(op_path, ID, opts, seq_len=1)

            for _ in tqdm(range(len(offline_cholec80_test))):
                data, target = next(offline_cholec80_test)

                if corrupt_mode != 0:
                    if isinstance(data, dict) and 'img' in data:
                        raw_frame = data['img']
                        corrupted_frame = apply_corruption(raw_frame, corrupt_mode)
                        data['img'] = corrupted_frame

                data, target = prepare_batch(data, target)

                if opts.shuffle:
                    model.net_short.temporal_head.reset()

                if opts.sliding_window:
                    output = model.forward_sliding_window(data)
                else:
                    output = model.forward(data)

                if isinstance(output, tuple):
                    output = output[0]
                output = [output[-1][:, -1:, :]]
                target = target[:, -1:]

                model.update_stats(0, output, target, mode='test')

                if opts.task == 'phase':
                    _, pred = output[-1].max(dim=2)
                    predictions.append(pred.flatten())
                    labels.append(target.flatten())

                elif opts.task == 'anticipation':
                    pred = output[-1][0]
                    pred *= opts.horizon
                    target *= opts.horizon
                    predictions.append(pred.flatten(end_dim=-2))
                    labels.append(target.flatten(end_dim=-2))

            predictions = torch.cat(predictions)
            labels = torch.cat(labels)

            if opts.task == 'phase':
                predictions = pd.DataFrame(predictions.cpu().numpy(), columns=['Phase'])
                labels = pd.DataFrame(labels.cpu().numpy(), columns=['Phase'])

            elif opts.task == 'anticipation':
                predictions = pd.DataFrame(predictions.cpu().numpy(),
                                           columns=['Bipolar', 'Scissors', 'Clipper', 'Irrigator', 'SpecBag'])
                labels = pd.DataFrame(labels.cpu().numpy(),
                                      columns=['Bipolar', 'Scissors', 'Clipper', 'Irrigator', 'SpecBag'])

            predictions.to_csv(os.path.join(pred_folder, f'video{ID}-phase.txt'),
                               index=True, index_label='Frame', sep='\t')
            labels.to_csv(os.path.join(gt_folder, f'video{ID}-phase.txt'),
                          index=True, index_label='Frame', sep='\t')
            print(f'Saved predictions/labels for video {ID} (corr={corrupt_mode})')

        epoch = torch.load(opts.resume)['epoch']
        model.summary(log_file=os.path.join(pred_folder, 'log.txt'), epoch=epoch)

    from visualization.Visualize import visual_main
    visual_main(out_folder, suffixpred=suffix[4:])
