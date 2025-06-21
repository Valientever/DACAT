#!/usr/bin/env python3
"""
eval.py

Compute per-video relaxed‐boundary metrics (accuracy, precision, recall, jaccard)
from stored ground-truth & prediction files, and write a tidy CSV.

Usage:
  python eval.py \
    --root_dir        /path/to/results \
    --experiment_name <exp_folder> \
    --predict_name    <predict_folder> \
    --output_csv      /path/to/output/metrics.csv \
    --train_condition <any_string> \
    --eval_condition  <any_string>
"""
import os
import argparse
import numpy as np
import pandas as pd
from scipy.ndimage import label

def read_phase_label(path):
    """Read two-column frame_index + phase_label file."""
    frames, phases = [], []
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2: continue
            frames.append(int(parts[0]))
            phases.append(parts[1])
    return np.array(frames), phases

def get_components(arr, phase_id):
    """Return list of connected-component index arrays where arr == phase_id."""
    mask = (arr == phase_id)
    labeled, n = label(mask.astype(int))
    return [np.where(labeled == i)[0] for i in range(1, n+1)]

def relaxed_metrics(gt, pred, fps=1):
    """
    Apply 10-second relaxed boundary, then compute per-phase:
    - jaccard, precision, recall, and overall accuracy.
    """
    t0 = 10 * fps
    diff = pred - gt
    upd = diff.copy()
    n = len(gt)
    for p in range(1, 8):
        for comp in get_components(gt, p):
            s, e = comp[0], comp[-1]
            seg = upd[s:e+1]
            t = min(t0, len(seg))
            if p in (4,5):
                seg[:t][seg[:t] == -1] = 0
                mask = np.isin(seg[-t:], [1,2])
                seg[-t:][mask] = 0
            elif p in (6,7):
                seg[:t][np.isin(seg[:t], [-1,-2])] = 0
                mask = np.isin(seg[-t:], [1,2])
                seg[-t:][mask] = 0
            else:
                seg[:t][seg[:t] == -1] = 0
                seg[-t:][seg[-t:] == 1] = 0
            upd[s:e+1] = seg
    j, pr, rc = [], [], []
    for p in range(1, 8):
        gtc = get_components(gt, p)
        predc = get_components(pred, p)
        if not gtc:
            j.append(np.nan); pr.append(np.nan); rc.append(np.nan)
            continue
        union = set().union(*gtc, *predc)
        union = np.array(list(union))
        tp = np.sum(upd[union] == 0)
        j.append(tp / len(union) * 100)
        sumP, sumG = np.sum(pred == p), np.sum(gt == p)
        pr.append((tp * 100 / sumP) if sumP > 0 else np.nan)
        rc.append(tp * 100 / sumG)
    acc = np.sum(upd == 0) / n * 100
    
    # Convert to numpy arrays for clamping
    j_arr = np.array(j)
    pr_arr = np.array(pr)
    rc_arr = np.array(rc)
    
    # **CRITICAL FIX**: Clamp values > 100% to 100%
    # This matches the MATLAB code behavior: index = find(prec>100); prec(index)=100;
    j_arr[j_arr > 100] = 100
    pr_arr[pr_arr > 100] = 100
    rc_arr[rc_arr > 100] = 100
    acc = min(acc, 100)  # Also clamp accuracy
    
    return j_arr, pr_arr, rc_arr, acc

def main():
    p = argparse.ArgumentParser(description="Per‑video relaxed metrics")
    p.add_argument('--root_dir',        required=True)
    p.add_argument('--experiment_name', required=True)
    p.add_argument('--predict_name',    required=True)
    p.add_argument('--output_csv',      required=True)
    p.add_argument('--train_condition', default=None,
                   help="override train_condition")
    p.add_argument('--eval_condition',  default=None,
                   help="override eval_condition")
    args = p.parse_args()

    # Prepare output folder
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    # Locate GT & prediction folders
    base = os.path.join(args.root_dir, args.experiment_name, args.predict_name)
    gt_dir   = os.path.join(base, 'gt')
    pred_dir = os.path.join(base, 'predv2_DACAT')

    # Determine conditions
    train_cond = args.train_condition or args.predict_name
    eval_cond  = args.eval_condition  or args.experiment_name
    if args.eval_condition is None and eval_cond.endswith('_w28'):
        eval_cond = eval_cond[:-4]

    records = []
    for vid in range(41, 81):
        gt_path   = os.path.join(gt_dir,   f"video{vid}-phase.txt")
        pred_path = os.path.join(pred_dir, f"video{vid}-phase.txt")
        if not os.path.isfile(gt_path) or not os.path.isfile(pred_path):
            print(f"⚠️  Missing video {vid}, skipping")
            continue

        gt_frames, gt_phases     = read_phase_label(gt_path)
        pred_frames, pred_phases = read_phase_label(pred_path)
        if not np.array_equal(gt_frames, pred_frames):
            raise ValueError(f"Frame mismatch in video{vid}")

        gtID   = np.array([int(x)+1 for x in gt_phases])
        predID = np.array([int(x)+1 for x in pred_phases])

        j, pr, rc, acc = relaxed_metrics(gtID, predID, fps=1)
        metrics = {
            'accuracy':  acc,
            'precision': np.nanmean(pr),
            'recall':    np.nanmean(rc),
            'jaccard':   np.nanmean(j)
        }
        for m, val in metrics.items():
            records.append({
                'video_id': vid,
                'train_condition': train_cond,
                'eval_condition':  eval_cond,
                'metric':          m,
                'score':           float(val)
            })

    df = pd.DataFrame.from_records(records,
        columns=['video_id','train_condition','eval_condition','metric','score']
    )
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote per‑video CSV to {args.output_csv}")

if __name__ == '__main__':
    import pandas as pd
    main()
