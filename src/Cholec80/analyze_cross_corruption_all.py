#!/usr/bin/env python3
"""
analyze_cross_corruption_all.py

Comprehensive cross-corruption statistical significance testing.
Uses the EXACT SAME Wilcoxon test implementation as analyze_test.py to ensure consistent p-values.
"""
import argparse
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from itertools import combinations
import sys

def parse_args():
    p = argparse.ArgumentParser(
        description="Comprehensive cross-corruption statistical analysis (using existing Wilcoxon implementation)"
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

def get_conditions_and_metrics(df):
    train_conditions = sorted(df['train_condition'].unique())
    eval_conditions = sorted(df['eval_condition'].unique())
    metrics = sorted(df['metric'].unique())
    
    print(f"Found {len(train_conditions)} training conditions: {train_conditions}")
    print(f"Found {len(eval_conditions)} evaluation conditions: {eval_conditions}")
    print(f"Found {len(metrics)} metrics: {metrics}")
    
    return train_conditions, eval_conditions, metrics

def wilcoxon_pairwise_test(vals1, vals2, alpha):
    """
    EXACT SAME Wilcoxon test implementation as analyze_test.py
    This ensures consistent p-values across all analyses.
    """
    # Use the exact same approach as analyze_test.py line 63-65
    med_1 = pd.Series(vals1).median()
    med_2 = pd.Series(vals2).median()
    diff = med_2 - med_1
    
    # Use the exact same Wilcoxon call as analyze_test.py line 64
    stat, p = wilcoxon(vals1, vals2)
    sig = p < alpha
    
    return {
        'median_1': med_1,
        'median_2': med_2, 
        'difference': diff,
        'wilcoxon_stat': stat,
        'p_value': p,
        'significance': sig
    }

def run_pairwise_tests(df, train_conditions, eval_conditions, metrics, alpha, logf):
    """
    Run pairwise comparisons using the EXACT SAME statistical method as analyze_test.py
    """
    
    logf.write("# Comprehensive Cross-Corruption Statistical Analysis\n")
    logf.write("# Using SAME Wilcoxon implementation as analyze_test.py\n")
    logf.write(f"Alpha = {alpha}\n")
    logf.write(f"Total training conditions: {len(train_conditions)}\n")
    logf.write(f"Total evaluation conditions: {len(eval_conditions)}\n")
    logf.write(f"Pairwise comparisons per eval condition: {len(list(combinations(train_conditions, 2)))}\n\n")

    all_results = []
    
    for eval_cond in eval_conditions:
        logf.write(f"{'='*60}\n")
        logf.write(f"EVALUATION CONDITION: {eval_cond}\n")
        logf.write(f"{'='*60}\n\n")
        
        # Get data for this evaluation condition - SAME as analyze_test.py
        eval_data = df[df['eval_condition'] == eval_cond]
        
        for metric in metrics:
            logf.write(f"--- Metric: {metric.upper()} ---\n")
            
            # Get data for this metric - SAME as analyze_test.py
            metric_data = eval_data[eval_data['metric'] == metric]
            
            # Pivot to get train conditions as columns - SAME as analyze_test.py
            pivot_data = metric_data.pivot(index='video_id', columns='train_condition', values='score')
            
            # Check which training conditions have data
            available_trains = [tc for tc in train_conditions if tc in pivot_data.columns and not pivot_data[tc].isna().all()]
            
            if len(available_trains) < 2:
                logf.write(f"  ⚠️  Insufficient data for comparisons (only {len(available_trains)} train conditions)\n\n")
                continue
            
            logf.write(f"  Available training conditions: {available_trains}\n")
            logf.write(f"  Pairwise comparisons: {len(list(combinations(available_trains, 2)))}\n\n")
            
            # Perform all pairwise comparisons using SAME method as analyze_test.py
            for train1, train2 in combinations(available_trains, 2):
                # Get values - SAME approach as analyze_test.py lines 61-62
                clean_vals = pivot_data[train1].values  # equivalent to clean_vals in analyze_test.py
                corr_vals = pivot_data[train2].values   # equivalent to corr_vals in analyze_test.py
                
                # Remove NaN values exactly like analyze_test.py would
                clean_vals = clean_vals[~pd.isna(clean_vals)]
                corr_vals = corr_vals[~pd.isna(corr_vals)]
                
                if len(clean_vals) < 5 or len(corr_vals) < 5:
                    logf.write(f"    {train1} vs {train2}: Insufficient data\n")
                    continue
                
                # Use EXACT SAME statistical calculation as analyze_test.py
                try:
                    stats_result = wilcoxon_pairwise_test(clean_vals, corr_vals, alpha)
                    
                    result = {
                        'eval_condition': eval_cond,
                        'metric': metric,
                        'train_condition_1': train1,
                        'train_condition_2': train2,
                        'median_1': stats_result['median_1'],
                        'median_2': stats_result['median_2'],
                        'difference': stats_result['difference'],
                        'effect_size': stats_result['difference'],
                        'pairs': len(clean_vals),  # Same as analyze_test.py line 66
                        'wilcoxon_stat': stats_result['wilcoxon_stat'],
                        'p_value': stats_result['p_value'],
                        'significance': stats_result['significance'],
                        'direction': 'train2_better' if stats_result['difference'] > 0 else 'train1_better' if stats_result['difference'] < 0 else 'neutral'
                    }
                    
                    all_results.append(result)
                    
                    # Log this comparison
                    direction_symbol = "📈" if stats_result['difference'] > 0 else "📉" if stats_result['difference'] < 0 else "➡️"
                    sig_symbol = "✅" if stats_result['significance'] else "❌"
                    
                    logf.write(f"    {train1} vs {train2}: {direction_symbol} {sig_symbol}\n")
                    logf.write(f"      Medians: {stats_result['median_1']:.4f} vs {stats_result['median_2']:.4f} (diff: {stats_result['difference']:+.4f})\n")
                    logf.write(f"      p-value: {stats_result['p_value']:.6f} (n={len(clean_vals)})\n")
                    
                except Exception as e:
                    logf.write(f"    {train1} vs {train2}: ERROR - {str(e)}\n")
                    continue
            
            logf.write("\n")
        
        logf.write("\n")
    
    return all_results

def generate_summary_statistics(results, logf):
    """Generate overall summary statistics"""
    
    if not results:
        logf.write("No results to summarize.\n")
        return
    
    df_results = pd.DataFrame(results)
    
    logf.write("="*80 + "\n")
    logf.write("OVERALL SUMMARY STATISTICS\n")
    logf.write("="*80 + "\n\n")
    
    # Overall statistics
    total_comparisons = len(df_results)
    significant_comparisons = df_results['significance'].sum()
    significance_rate = significant_comparisons / total_comparisons * 100
    
    logf.write(f"Total pairwise comparisons: {total_comparisons}\n")
    logf.write(f"Significant differences: {significant_comparisons}\n")
    logf.write(f"Overall significance rate: {significance_rate:.1f}%\n\n")
    
    # By evaluation condition
    logf.write("Significance by Evaluation Condition:\n")
    eval_summary = df_results.groupby('eval_condition').agg({
        'significance': ['count', 'sum', 'mean']
    }).round(3)
    eval_summary.columns = ['Total_Comparisons', 'Significant', 'Significance_Rate']
    logf.write(eval_summary.to_string())
    logf.write("\n\n")
    
    # By metric
    logf.write("Significance by Metric:\n")
    metric_summary = df_results.groupby('metric').agg({
        'significance': ['count', 'sum', 'mean']
    }).round(3)
    metric_summary.columns = ['Total_Comparisons', 'Significant', 'Significance_Rate']
    logf.write(metric_summary.to_string())
    logf.write("\n\n")
    
    # Most significant differences
    logf.write("Most Significant Differences (p < 0.001):\n")
    highly_significant = df_results[df_results['p_value'] < 0.001].sort_values('p_value')
    if len(highly_significant) > 0:
        for _, row in highly_significant.head(10).iterrows():
            direction = "↑" if row['difference'] > 0 else "↓"
            logf.write(f"  {row['train_condition_1']} vs {row['train_condition_2']} on {row['eval_condition']} ({row['metric']})\n")
            logf.write(f"    {direction} Effect: {row['difference']:+.4f}, p={row['p_value']:.2e}\n")
    else:
        logf.write("  No highly significant differences found.\n")
    logf.write("\n")

def main():
    args = parse_args()
    df = load_and_validate(args.input)
    train_conditions, eval_conditions, metrics = get_conditions_and_metrics(df)

    print(f"Starting comprehensive cross-corruption analysis...")
    print(f"Using EXACT SAME Wilcoxon implementation as analyze_test.py for consistency")
    print(f"This will perform {len(train_conditions)*(len(train_conditions)-1)/2} pairwise comparisons")
    print(f"across {len(eval_conditions)} evaluation conditions and {len(metrics)} metrics")
    
    with open(args.log, 'w') as logf:
        results = run_pairwise_tests(df, train_conditions, eval_conditions, metrics, args.alpha, logf)
        generate_summary_statistics(results, logf)
    
    print(f"✅ Comprehensive analysis complete!")
    print(f"📄 Results saved to: {args.log}")
    print(f"📊 Total pairwise comparisons: {len(results)}")
    
    if results:
        significant = sum(1 for r in results if r['significance'])
        print(f"🔍 Significant differences found: {significant}/{len(results)} ({significant/len(results)*100:.1f}%)")

if __name__ == '__main__':
    main()
