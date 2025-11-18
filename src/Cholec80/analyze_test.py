#!/usr/bin/env python3
"""
analyze_train_vs_baseline.py

Automated paired significance testing comparing:
  - Model trained on clean data (baseline)
  - Model trained on each corrupted dataset

For each corruption & each metric, runs a Wilcoxon signed‑rank test across all videos,
and logs a table with columns:

  metric | pairs | median | train_condition | eval_condition | difference |
  wilcoxon_stat | p-value | significance | ci_lower | ci_upper

95% Confidence Intervals calculated using bootstrap method.

Usage:
  python analyze_train_vs_baseline.py \
    --input scores.csv \
    --log   results_log.txt \
    [--alpha 0.05] \
    [--bootstrap_iterations 10000]
"""
import argparse
import pandas as pd
import numpy as np
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
    p.add_argument('-b','--bootstrap_iterations', type=int, default=10000,
                   help='Number of bootstrap iterations for CI calculation (default 10000)')
    p.add_argument('--ci_level', type=float, default=0.95,
                   help='Confidence interval level (default 0.95)')
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

def bootstrap_ci_for_paired_difference(clean_vals, corr_vals, n_iterations=10000, ci_level=0.95):
    """
    Calculate bootstrap confidence interval for paired differences.
    
    Parameters:
    - clean_vals: array of baseline (clean-trained) scores
    - corr_vals: array of corruption-trained scores
    - n_iterations: number of bootstrap samples
    - ci_level: confidence level (default 0.95 for 95% CI)
    
    Returns:
    - (ci_lower, ci_upper): confidence interval bounds for the difference
    """
    n = len(clean_vals)
    differences = corr_vals - clean_vals
    
    # Store bootstrap difference medians
    boot_diff_medians = []
    
    np.random.seed(42)  # For reproducibility
    
    for _ in range(n_iterations):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        boot_clean = clean_vals[indices]
        boot_corr = corr_vals[indices]
        boot_diff = boot_corr - boot_clean
        
        # Calculate median of differences
        boot_diff_medians.append(np.median(boot_diff))
    
    # Calculate percentile-based CI
    alpha = 1 - ci_level
    ci_lower = np.percentile(boot_diff_medians, 100 * alpha / 2)
    ci_upper = np.percentile(boot_diff_medians, 100 * (1 - alpha / 2))
    
    return ci_lower, ci_upper

def run_tests(df, corruptions, metrics, alpha, logf, bootstrap_iterations=10000, ci_level=0.95):
    logf.write("# Train vs Baseline Significance Analysis\n")
    logf.write(f"Alpha = {alpha}\n")
    logf.write(f"Bootstrap iterations = {bootstrap_iterations}\n")
    logf.write(f"Confidence interval level = {ci_level*100:.0f}%\n\n")

    for corr in corruptions:
        subset = df[df['eval_condition']==corr]
        rows = []
        for m in metrics:
            sub = subset[subset['metric']==m]
            wide = sub.pivot(index='video_id', columns='train_condition', values='score')
            clean_vals = wide['clean'].values
            corr_vals  = wide[corr].values

            med_clean = pd.Series(clean_vals).median()
            med_corr = pd.Series(corr_vals).median()
            # Calculate difference as median of paired differences (consistent with bootstrap CI)
            paired_diffs = corr_vals - clean_vals
            diff = np.median(paired_diffs)
            stat, p  = wilcoxon(clean_vals, corr_vals)
            sig      = p < alpha
            
            # Calculate bootstrap confidence interval
            ci_lower, ci_upper = bootstrap_ci_for_paired_difference(
                clean_vals, corr_vals, 
                n_iterations=bootstrap_iterations,
                ci_level=ci_level
            )

            rows.append({
                'metric':          m,
                'pairs':           len(clean_vals),
                'median_baseline': med_clean,
                'median_trained':  med_corr,
                'train_condition': corr,
                'eval_condition':  corr,
                'difference':      diff,
                'ci_lower':        ci_lower,
                'ci_upper':        ci_upper,
                'wilcoxon_stat':   stat,
                'p-value':         p,
                'significance':    sig
            })

        table = pd.DataFrame(rows, columns=[
            'metric','pairs','median_baseline','median_trained','train_condition','eval_condition',
            'difference','ci_lower','ci_upper','wilcoxon_stat','p-value','significance'
        ])
        logf.write(f"=== Corruption: {corr} ===\n")
        logf.write(table.to_string(index=False, float_format="%.4f"))
        logf.write("\n\n")

def main():
    args = parse_args()
    df = load_and_validate(args.input)
    corruptions, metrics = detect_corruptions_and_metrics(df)

    print(f"🔬 Running statistical analysis with {args.bootstrap_iterations} bootstrap iterations...")
    print(f"📊 Calculating {args.ci_level*100:.0f}% confidence intervals...")
    
    with open(args.log,'w') as logf:
        run_tests(df, corruptions, metrics, args.alpha, logf, 
                 args.bootstrap_iterations, args.ci_level)
    print(f"✅ Analysis complete. Log saved to {args.log}")

if __name__=='__main__':
    main()
