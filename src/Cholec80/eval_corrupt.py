import os
import numpy as np
from scipy.ndimage import label
from ipdb import set_trace
import pandas as pd

# -----------------------------
# Your original read/eval code:
# -----------------------------
def read_phase_label(filename):
    """
    Reads a two-column text file of the form:
        frame_index   phase_label
    Returns a tuple (frames, phases), where
        frames is a 1D NumPy array of frame indices,
        phases is a list of phase labels (strings).
    """
    frames = []
    phase_labels = []
    with open(filename, 'r') as f:
        hearder = f.readline()  # changed from 'header' to fix a possible typo

        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            frames.append(int(parts[0]))
            phase_labels.append(parts[1])
    return np.array(frames), phase_labels


def get_connected_components_1d(label_array, phase_id):
    mask = (label_array == phase_id)
    labeled, num_features = label(mask.astype(int))
    components = []
    for comp_id in range(1, num_features + 1):
        comp_indices = np.where(labeled == comp_id)[0]
        components.append(comp_indices)
    return components


def evaluate(gtLabelID, predLabelID, fps):
    oriT = 10 * fps
    diff = predLabelID - gtLabelID
    updatedDiff = diff.copy()
    n_frames = len(gtLabelID)

    for iPhase in range(1, 8):
        gt_conn = get_connected_components_1d(gtLabelID, iPhase)
        for comp_indices in gt_conn:
            start_idx = comp_indices[0]
            end_idx = comp_indices[-1]
            curDiff = updatedDiff[start_idx:end_idx+1]

            t = oriT
            if t > len(curDiff):
                t = len(curDiff)
                print(f"Very short phase {iPhase}")

            if iPhase == 4 or iPhase == 5:
                left_end = curDiff[:t]
                left_mask = (left_end == -1)
                curDiff[:t][left_mask] = 0

                right_end = curDiff[-t:]
                right_mask = np.isin(right_end, [1, 2])
                curDiff[-t:][right_mask] = 0

            elif iPhase == 6 or iPhase == 7:
                left_end = curDiff[:t]
                left_mask = np.isin(left_end, [-1, -2])
                curDiff[:t][left_mask] = 0

                right_end = curDiff[-t:]
                right_mask = np.isin(right_end, [1, 2])
                curDiff[-t:][right_mask] = 0
            else:
                left_end = curDiff[:t]
                left_mask = (left_end == -1)
                curDiff[:t][left_mask] = 0

                right_end = curDiff[-t:]
                right_mask = (right_end == 1)
                curDiff[-t:][right_mask] = 0

            updatedDiff[start_idx:end_idx+1] = curDiff

    res = []
    prec = []
    rec = []

    for iPhase in range(1, 8):
        gt_conn = get_connected_components_1d(gtLabelID, iPhase)
        pred_conn = get_connected_components_1d(predLabelID, iPhase)
        if len(gt_conn) == 0:
            res.append(np.nan)
            prec.append(np.nan)
            rec.append(np.nan)
            continue
        gt_indices = set()
        for comp in gt_conn:
            gt_indices.update(comp)
        pred_indices = set()
        for comp in pred_conn:
            pred_indices.update(comp)
        iPUnion = np.array(list(gt_indices.union(pred_indices)))
        tp = np.sum(updatedDiff[iPUnion] == 0)
        jaccard = (tp / len(iPUnion)) * 100
        res.append(jaccard)

        sumPred = np.sum(predLabelID == iPhase)
        sumGT   = np.sum(gtLabelID == iPhase)
        sumTP   = tp

        if sumPred == 0:
            phase_prec = np.nan
        else:
            phase_prec = (sumTP * 100.0 / sumPred)
        phase_rec = (sumTP * 100.0 / sumGT)
        prec.append(phase_prec)
        rec.append(phase_rec)

    acc = (np.sum(updatedDiff == 0) / n_frames) * 100.0

    return np.array(res), np.array(prec), np.array(rec), acc


# -------------------------------------------------------------
# NEW MAIN FUNCTION with a new parameter "corrupt_pred"
# to evaluate multiple corruption subfolders in one run.
# -------------------------------------------------------------
def main():
    import math

    # Root directory
    maindir = r"/home/santhi/Documents/DACAT/predictions_corrupt"

    # frames per second
    fps = 1

    # Phase names
    phases = [
        'Preparation', 
        'CalotTriangleDissection',
        'ClippingCutting',
        'GallbladderDissection',
        'GallbladderPackaging',
        'CleaningCoagulation',
        'GallbladderRetraction'
    ]

    # ------------------------------------------------------------------------
    # NEW: Suppose we have subfolders for predictions, e.g.:
    #  'predv2_DACAT_corrupt_0', 'predv2_DACAT_corrupt_1', ..., 'predv2_DACAT_corrupt_6'
    # We'll store them in a list. Also keep the original 'predv2_DACAT' if needed.
    # In practice, you might read these from command-line or config.
    # ------------------------------------------------------------------------
    corrupt_preds = [
        'predv2_DACAT_corrupt_0',
        'predv2_DACAT_corrupt_1',
        'predv2_DACAT_corrupt_2',
        'predv2_DACAT_corrupt_3',
        'predv2_DACAT_corrupt_4',
        'predv2_DACAT_corrupt_5',
        'predv2_DACAT_corrupt_6'
    ]
    # If you also want the "original" (no corruption) predictions stored under
    # 'predv2_DACAT', just add it to the list:
    # corrupt_preds.insert(0, 'predv2_DACAT')  # optional

    # Gather ground-truth text file paths
    gt_root_folder = os.path.join(maindir, "gt_corrupt")
    phase_ground_truths = []
    for k in range(41, 81):
        phase_ground_truths.append(
            os.path.join(gt_root_folder, f"video{k}-phase.txt")
        )

    # We'll accumulate results in a dictionary:
    all_results = {}  # key = corrupt folder name, value = (acc, jacc, prec, rec, etc.)

    for corrupt_name in corrupt_preds:
        print(f"\n=== Evaluating predictions in folder: {corrupt_name} ===")
        # We do exactly the same logic as your original code, but read from this folder

        # Arrays to accumulate metrics
        jaccard_list = []
        prec_list = []
        rec_list = []
        acc_list = []

        # For each GT file, find the corresponding prediction
        for gt_file in phase_ground_truths:
            basename = os.path.basename(gt_file)  # e.g. "video41-phase.txt"
            # Build path to the corrupted predictions folder
            predroot = os.path.join(maindir, corrupt_name)
            pred_file = os.path.join(predroot, basename)

            if not os.path.isfile(pred_file):
                # print(f"Warning: No prediction file found for {basename} in {corrupt_name}")
                continue

            # Read GT and predictions
            gt_frames, gt_phases = read_phase_label(gt_file)
            pred_frames, pred_phases = read_phase_label(pred_file)

            if (len(gt_frames) != len(pred_frames)):
                raise ValueError(
                    f"ERROR: {gt_file}\n"
                    f"Ground truth and prediction have different sizes"
                )
            if not np.array_equal(gt_frames, pred_frames):
                raise ValueError(
                    f"ERROR: {gt_file}\n"
                    f"The frame index in ground truth and prediction is not equal"
                )

            gtLabelID = np.zeros_like(gt_frames, dtype=int)
            predLabelID = np.zeros_like(gt_frames, dtype=int)
            for j in range(7):  # j=0..6
                phase_str = str(j)
                gtLabelID[np.where(np.array(gt_phases) == phase_str)] = j + 1
                predLabelID[np.where(np.array(pred_phases) == phase_str)] = j + 1

            # Evaluate
            res, prec, rec_, acc = evaluate(gtLabelID, predLabelID, fps)
            jaccard_list.append(res)
            prec_list.append(prec)
            rec_list.append(rec_)
            acc_list.append(acc)

        if len(jaccard_list) == 0:
            print(f"No predictions found in {corrupt_name}, skipping stats.")
            continue

        jaccard_arr = np.vstack(jaccard_list).T
        prec_arr = np.vstack(prec_list).T
        rec_arr = np.vstack(rec_list).T
        acc_arr = np.array(acc_list)

        jaccard_arr[jaccard_arr > 100] = 100
        prec_arr[prec_arr > 100] = 100
        rec_arr[rec_arr > 100] = 100

        mean_jacc_per_phase = np.nanmean(jaccard_arr, axis=1)
        mean_prec_per_phase = np.nanmean(prec_arr, axis=1)
        mean_rec_per_phase  = np.nanmean(rec_arr, axis=1)

        std_jacc_per_phase = np.nanstd(jaccard_arr, axis=1)
        std_prec_per_phase = np.nanstd(prec_arr, axis=1)
        std_rec_per_phase  = np.nanstd(rec_arr, axis=1)

        mean_jacc = np.nanmean(mean_jacc_per_phase) 
        std_jacc  = np.nanstd(mean_jacc_per_phase)
        mean_prec = np.nanmean(mean_prec_per_phase)
        std_prec  = np.nanstd(mean_prec_per_phase)
        mean_rec  = np.nanmean(mean_rec_per_phase)
        std_rec   = np.nanstd(mean_rec_per_phase)
        mean_acc = np.mean(acc_arr)
        std_acc  = np.std(acc_arr)

        # Store the summary in our dictionary
        all_results[corrupt_name] = {
            'mean_acc': mean_acc,
            'std_acc': std_acc,
            'mean_prec': mean_prec,
            'std_prec': std_prec,
            'mean_rec': mean_rec,
            'std_rec': std_rec,
            'mean_jacc': mean_jacc,
            'std_jacc': std_jacc
        }

        # Print short summary
        print("\n================================================")
        print(f"{'Phase':>25s} | {'Jacc':>6s} | {'Prec':>6s} | {'Rec':>6s} |")
        print("================================================")
        for iPhase, phase_name in enumerate(phases):
            print(
                f"{phase_name:>25s} | "
                f"{mean_jacc_per_phase[iPhase]:6.2f} | "
                f"{mean_prec_per_phase[iPhase]:6.2f} | "
                f"{mean_rec_per_phase[iPhase]:6.2f} |"
            )
            print("---------------------------------------------")
        print("================================================")
        print(f" Corruption folder: {corrupt_name}")
        print(f" Mean accuracy:  {mean_acc:5.2f} +- {std_acc:5.2f}")
        print(f" Mean precision: {mean_prec:5.2f} +- {std_prec:5.2f}")
        print(f" Mean recall:    {mean_rec:5.2f} +- {std_rec:5.2f}")
        print(f" Mean jaccard:   {mean_jacc:5.2f} +- {std_jacc:5.2f}\n")

    # --------------------------------------------------------------------
    # Finally, produce a "tabular" output comparing all corruptions
    # side by side. For brevity, we only show mean_acc, mean_prec, mean_rec,
    # mean_jacc. Feel free to add the std columns or more detail.
    # --------------------------------------------------------------------
    print("\n\n=== COMPARISON ACROSS CORRUPTION FOLDERS ===\n")
    print(f"{'Corruption':>20s} | {'Acc':>6s} | {'Prec':>6s} | {'Rec':>6s} | {'Jacc':>6s}")
    print("-----------------------------------------------------------------")
    for c_name, res in all_results.items():
        print(f"{c_name:>20s} | "
              f"{res['mean_acc']:6.2f} | "
              f"{res['mean_prec']:6.2f} | "
              f"{res['mean_rec']:6.2f} | "
              f"{res['mean_jacc']:6.2f} ")
    print("-----------------------------------------------------------------")


if __name__ == "__main__":
    main()
