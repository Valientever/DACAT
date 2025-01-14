import os
import numpy as np
from scipy.ndimage import label
from ipdb import set_trace

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
        hearder = f.readline()

        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # set_trace()
            frames.append(int(parts[0]))
            phase_labels.append(parts[1])
    return np.array(frames), phase_labels


def get_connected_components_1d(label_array, phase_id):
    """
    Replicates the bwconncomp(gtLabelID == iPhase) logic in MATLAB
    for a 1D array of labels. Returns a list of arrays, where each
    array holds the indices of one connected component.
    """
    # mask = True where the label matches `phase_id`
    mask = (label_array == phase_id)
    # label() will identify connected components in 1D (size x1)
    labeled, num_features = label(mask.astype(int))
    
    components = []
    for comp_id in range(1, num_features + 1):
        # indices of the current connected component
        comp_indices = np.where(labeled == comp_id)[0]
        components.append(comp_indices)
    
    return components


def evaluate(gtLabelID, predLabelID, fps):
    """
    Replicates the Evaluate.m function in MATLAB.
    Computes:
      - Jaccard index (res) per phase with relaxed boundaries
      - Precision (prec) per phase with relaxed boundaries
      - Recall (rec) per phase with relaxed boundaries
      - Accuracy (acc) with relaxed boundaries over the entire sequence

    gtLabelID, predLabelID : 1D NumPy arrays of phase IDs (1..7)
    fps                    : frames per second (integer)
    """
    # Relaxed boundary for 10 seconds
    oriT = 10 * fps
    
    # diff array: pred - gt
    diff = predLabelID - gtLabelID
    
    # We'll build updatedDiff so that it matches the logic in Evaluate.m
    updatedDiff = diff.copy()  # start with a copy
    n_frames = len(gtLabelID)

    # For each of the 7 phases, we find connected components in GT
    # then adjust diff in those segments according to the relaxed boundary
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

            # Apply the relaxed boundary logic depending on the phase
            if iPhase == 4 or iPhase == 5:
                # GallbladderDissection (4) or GallbladderPackaging (5)
                # "late transition"
                #   if curDiff(1:t)==-1 -> set those to 0
                left_end = curDiff[:t]
                left_mask = (left_end == -1)
                curDiff[:t][left_mask] = 0

                # "early transition"
                #   if curDiff(end-t+1:end)==1 or ==2 -> set those to 0
                right_end = curDiff[-t:]
                right_mask = np.isin(right_end, [1, 2])
                curDiff[-t:][right_mask] = 0

            elif iPhase == 6 or iPhase == 7:
                # GallbladderDissection might jump (6 or 7)
                # "late transition"
                left_end = curDiff[:t]
                left_mask = np.isin(left_end, [-1, -2])
                curDiff[:t][left_mask] = 0

                # "early transition"
                right_end = curDiff[-t:]
                right_mask = np.isin(right_end, [1, 2])
                curDiff[-t:][right_mask] = 0
            else:
                # general situation
                left_end = curDiff[:t]
                left_mask = (left_end == -1)
                curDiff[:t][left_mask] = 0

                right_end = curDiff[-t:]
                right_mask = (right_end == 1)
                curDiff[-t:][right_mask] = 0

            updatedDiff[start_idx:end_idx+1] = curDiff

    # Now compute Jaccard, Precision, Recall for each phase
    res = []
    prec = []
    rec = []

    for iPhase in range(1, 8):
        # get connected components in GT and in pred for iPhase
        gt_conn = get_connected_components_1d(gtLabelID, iPhase)
        pred_conn = get_connected_components_1d(predLabelID, iPhase)

        # If no ground truth for iPhase, fill with NaNs
        if len(gt_conn) == 0:
            res.append(np.nan)
            prec.append(np.nan)
            rec.append(np.nan)
            continue
        
        # Build the union of all indices that belong to GT or Pred for iPhase
        # to replicate iPUnion
        gt_indices = set()
        for comp in gt_conn:
            gt_indices.update(comp)
        pred_indices = set()
        for comp in pred_conn:
            pred_indices.update(comp)

        iPUnion = np.array(list(gt_indices.union(pred_indices)))

        # True positives are positions in iPUnion where updatedDiff == 0
        # => means pred == gt for those frames
        tp = np.sum(updatedDiff[iPUnion] == 0)

        # Jaccard
        jaccard = (tp / len(iPUnion)) * 100
        res.append(jaccard)

        # Precision & Recall
        sumPred = np.sum(predLabelID == iPhase)
        sumGT   = np.sum(gtLabelID == iPhase)
        sumTP   = tp

        if sumPred == 0:
            # If there's no prediction for this phase, precision is NaN
            phase_prec = np.nan
        else:
            phase_prec = (sumTP * 100.0 / sumPred)

        phase_rec = (sumTP * 100.0 / sumGT)
        prec.append(phase_prec)
        rec.append(phase_rec)

    # Compute accuracy (relaxed)
    acc = (np.sum(updatedDiff == 0) / n_frames) * 100.0

    return np.array(res), np.array(prec), np.array(rec), acc


def main():
    # ------------------------------------------------------------------------
    # Equivalent to main.m
    # ------------------------------------------------------------------------
    import math

    # Root directory
    # maindir = r"D:/MATLAB/PhaseReg/20240704_convnext2"
    maindir = r"/home/santhi/Documents/DACAT/checkpoints"

    # Gather ground-truth text file paths
    # For example, we pick from 41..80
    phase_ground_truths = []
    gt_root_folder = os.path.join(maindir, "gt")
    for k in range(41, 81):
        phase_ground_truths.append(
            os.path.join(gt_root_folder, f"video{k}-phase.txt")
        )

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

    # frames per second
    fps = 1

    # Arrays to accumulate metrics
    # We have 7 phases, so jaccard, prec, rec will be 7 x Nvideos
    # We'll store them in lists first
    jaccard_list = []
    prec_list = []
    rec_list = []
    acc_list = []

    # Process each ground-truth file
    for gt_file in phase_ground_truths:
        # Build path to corresponding prediction
        # predroot = maindir + "/predv2_DACAT/"
        predroot = os.path.join(maindir, "predv2_DACAT")

        # e.g. "video41-phase.txt" -> "video41-phase.txt"
        basename = os.path.basename(gt_file)  # "video41-phase.txt"
        # For predictions
        pred_file = os.path.join(predroot, basename)
        # set_trace()
        # Read GT and predictions
        gt_frames, gt_phases = read_phase_label(gt_file)
        pred_frames, pred_phases = read_phase_label(pred_file)

        # Basic checks
        if (len(gt_frames) != len(pred_frames)):
            raise ValueError(
                f"ERROR: {gt_file}\n"
                f"Ground truth and prediction have different sizes"
            )
        # Check frame index consistency
        if not np.array_equal(gt_frames, pred_frames):
            raise ValueError(
                f"ERROR: {gt_file}\n"
                f"The frame index in ground truth and prediction is not equal"
            )

        # Reassign phase labels to numeric IDs (1..7)
        # In the original code, '0' -> 1, '1' -> 2, ..., '6' -> 7
        # Phase label strings in ground truth/pred are '0', '1', '2', ...
        # We'll map those strings to integers 1..7
        gtLabelID = np.zeros_like(gt_frames, dtype=int)
        predLabelID = np.zeros_like(gt_frames, dtype=int)
        for j in range(7):  # j=0..6
            phase_str = str(j)  # '0', '1', ...
            # Indices where gt_phases == j => label j+1
            gtLabelID[np.where(np.array(gt_phases) == phase_str)] = j + 1
            predLabelID[np.where(np.array(pred_phases) == phase_str)] = j + 1

        # Evaluate
        res, prec, rec, acc = evaluate(gtLabelID, predLabelID, fps)
        jaccard_list.append(res)
        prec_list.append(prec)
        rec_list.append(rec)
        acc_list.append(acc)

    # Convert to NumPy for easy stats: shape => (Nvideos, 7)
    jaccard_arr = np.vstack(jaccard_list).T  # shape => (7, Nvideos)
    prec_arr = np.vstack(prec_list).T        # shape => (7, Nvideos)
    rec_arr = np.vstack(rec_list).T          # shape => (7, Nvideos)
    acc_arr = np.array(acc_list)             # shape => (Nvideos,)

    # Post-processing: clamp > 100 to 100
    jaccard_arr[jaccard_arr > 100] = 100
    prec_arr[prec_arr > 100] = 100
    rec_arr[rec_arr > 100] = 100

    # Compute means and std dev (phase-wise and video-wise)
    mean_jacc_per_phase = np.nanmean(jaccard_arr, axis=1)   # shape => (7,)
    mean_prec_per_phase = np.nanmean(prec_arr, axis=1)      # shape => (7,)
    mean_rec_per_phase  = np.nanmean(rec_arr, axis=1)       # shape => (7,)

    std_jacc_per_phase = np.nanstd(jaccard_arr, axis=1)
    std_prec_per_phase = np.nanstd(prec_arr, axis=1)
    std_rec_per_phase  = np.nanstd(rec_arr, axis=1)

    # AVERAGE across all phases
    mean_jacc = np.nanmean(mean_jacc_per_phase) 
    std_jacc  = np.nanstd(mean_jacc_per_phase)
    mean_prec = np.nanmean(mean_prec_per_phase)
    std_prec  = np.nanstd(mean_prec_per_phase)
    mean_rec  = np.nanmean(mean_rec_per_phase)
    std_rec   = np.nanstd(mean_rec_per_phase)

    mean_acc = np.mean(acc_arr)
    std_acc  = np.std(acc_arr)

    # Display the results
    print("================================================")
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
    print(f"Mean accuracy:  {mean_acc:5.2f} +- {std_acc:5.2f}")
    print(f"Mean precision: {mean_prec:5.2f} +- {std_prec:5.2f}")
    print(f"Mean recall:    {mean_rec:5.2f} +- {std_rec:5.2f}")
    print(f"Mean jaccard:   {mean_jacc:5.2f} +- {std_jacc:5.2f}")


if __name__ == "__main__":
    main()
