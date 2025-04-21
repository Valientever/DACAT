#!/usr/bin/env python3
"""
analyze_train_vs_baseline.py

Automated paired significance testing comparing:
  - Model trained on clean data (baseline)
  - Model trained on each corrupted dataset

For each corruption & each metric, runs a Wilcoxon signed‑rank test across all videos,
and logs a table with columns:

  metric | pairs | median | train_condition | eval_condition | difference |
  wilcoxon_stat | p-value | significance

Usage:
  python analyze_train_vs_baseline.py \
    --input scores.csv \
    --log   results_log.txt \
    [--alpha 0.05]
"""
import argparse
import pandas as pd
from scipy.stats import wilcoxon
import sys

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare clean‑trained vs corruption‑trained on corrupted data"
    )
    p.add_argument('-i','--input', required=True,
                   help='CSV with columns: video_id,train_condition,eval_condition,metric,score')
    p.add_argument('-l','--log',   required=True,
                   help='Output log text file path')
    p.add_argument('-a','--alpha', type=float, default=0.05,
                   help='Significance level (default 0.05)')
    return p.parse_args()

def load_and_validate(path):
    df = pd.read_csv(path)
    req = {'video_id','train_condition','eval_condition','metric','score'}
    if not req.issubset(df.columns):
        print(f"ERROR: CSV must contain columns: {req}")
        sys.exit(1)
    return df

def detect_corruptions_and_metrics(df):
    metrics = sorted(df['metric'].unique())
    eval_conds = sorted(df['eval_condition'].unique())
    if 'clean' in eval_conds:
        eval_conds.remove('clean')
    return eval_conds, metrics

def run_tests(df, corruptions, metrics, alpha, logf):
    logf.write("# Train vs Baseline Significance Analysis\n")
    logf.write(f"Alpha = {alpha}\n\n")

    for corr in corruptions:
        subset = df[df['eval_condition']==corr]
        rows = []
        for m in metrics:
            sub = subset[subset['metric']==m]
            wide = sub.pivot(index='video_id', columns='train_condition', values='score')
            clean_vals = wide['clean'].values
            corr_vals  = wide[corr].values

            med_corr = pd.Series(corr_vals).median()
            diff     = med_corr - pd.Series(clean_vals).median()
            stat, p  = wilcoxon(clean_vals, corr_vals)
            sig      = p < alpha

            rows.append({
                'metric':          m,
                'pairs':           len(clean_vals),
                'median':          med_corr,
                'train_condition': corr,
                'eval_condition':  corr,
                'difference':      diff,
                'wilcoxon_stat':   stat,
                'p-value':         p,
                'significance':    sig
            })

        table = pd.DataFrame(rows, columns=[
            'metric','pairs','median','train_condition','eval_condition',
            'difference','wilcoxon_stat','p-value','significance'
        ])
        logf.write(f"=== Corruption: {corr} ===\n")
        logf.write(table.to_string(index=False, float_format="%.4f"))
        logf.write("\n\n")

def main():
    args = parse_args()
    df = load_and_validate(args.input)
    corruptions, metrics = detect_corruptions_and_metrics(df)

    with open(args.log,'w') as logf:
        run_tests(df, corruptions, metrics, args.alpha, logf)
    print(f"Analysis complete. Log saved to {args.log}")

if __name__=='__main__':
    main()
