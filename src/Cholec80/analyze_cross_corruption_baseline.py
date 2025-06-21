#!/usr/bin/env python3
"""
analyze_cross_corruption_baseline.py

Statistical analysis comparing corruption-trained models vs clean baseline.
This matches the format shown in the user's tables where each corruption type
is compared against the clean-trained model as a baseline.
"""
import argparse
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import sys
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(
        description="Corruption vs Clean baseline statistical analysis"
    )
    p.add_argument('-i','--input', default='results/all_per_video_metrics.csv',
                   help='CSV with columns: video_id,train_condition,eval_condition,metric,score')
    p.add_argument('-o','--output', default='results/corrected_analysis',
                   help='Output directory for results')
    p.add_argument('-a','--alpha', type=float, default=0.05,
                   help='Significance level (default 0.05)')
    return p.parse_args()

def load_and_validate(path):
    if not Path(path).exists():
        print(f"ERROR: Input file not found: {path}")
        sys.exit(1)
    
    df = pd.read_csv(path)
    req = {'video_id','train_condition','eval_condition','metric','score'}
    if not req.issubset(df.columns):
        print(f"ERROR: CSV must contain columns: {req}")
        sys.exit(1)
    return df

def analyze_corruption_vs_clean(df, alpha, output_dir):
    """
    Compare corruption-trained models vs clean baseline on corresponding evaluation datasets.
    This matches the analysis shown in the user's tables.
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get corruption conditions (exclude clean)
    all_conditions = sorted(df['train_condition'].unique())
    corruption_conditions = [c for c in all_conditions if c != 'clean']
    metrics = sorted(df['metric'].unique())
    
    print(f"Found corruption conditions: {corruption_conditions}")
    print(f"Found metrics: {metrics}")
    
    results = []
    
    with open(output_dir / 'baseline_analysis_log.txt', 'w') as logf:
        logf.write("CORRUPTION vs CLEAN BASELINE ANALYSIS\n")
        logf.write("="*50 + "\n\n")
        logf.write(f"Alpha level: {alpha}\n")
        logf.write(f"Comparison: Each corruption-trained model vs clean-trained model\n")
        logf.write(f"Evaluation: On corresponding corruption type datasets\n\n")
        
        for corruption in corruption_conditions:
            logf.write(f"\n{'='*40}\n")
            logf.write(f"CORRUPTION TYPE: {corruption.upper()}\n")
            logf.write(f"{'='*40}\n\n")
            
            # Compare corruption-trained vs clean-trained on corruption evaluation dataset
            eval_subset = df[df['eval_condition'] == corruption]
            
            for metric in metrics:
                logf.write(f"--- Metric: {metric.upper()} ---\n")
                
                metric_subset = eval_subset[eval_subset['metric'] == metric]
                
                # Pivot to get training conditions as columns
                pivot_data = metric_subset.pivot(
                    index='video_id', 
                    columns='train_condition', 
                    values='score'
                ).dropna()
                
                # Check if we have both clean and corruption data
                if 'clean' not in pivot_data.columns or corruption not in pivot_data.columns:
                    logf.write(f"  ⚠️  Missing data for comparison\n")
                    continue
                
                # Get paired values for videos present in both conditions
                clean_values = pivot_data['clean']
                corruption_values = pivot_data[corruption]
                
                # Find common videos
                common_videos = clean_values.index.intersection(corruption_values.index)
                
                if len(common_videos) < 5:
                    logf.write(f"  ⚠️  Insufficient paired data: {len(common_videos)} videos\n")
                    continue
                
                # Extract paired values
                clean_paired = clean_values.loc[common_videos].values
                corruption_paired = corruption_values.loc[common_videos].values
                
                # Calculate statistics
                median_clean = np.median(clean_paired)
                median_corruption = np.median(corruption_paired)
                difference = median_corruption - median_clean  # positive means corruption is better
                
                # Wilcoxon signed-rank test
                try:
                    stat, p_value = wilcoxon(clean_paired, corruption_paired)
                    significance = p_value < alpha
                    
                    result = {
                        'corruption': corruption,
                        'metric': metric,
                        'eval_condition': corruption,  # Evaluated on corruption dataset
                        'n_videos': len(common_videos),
                        'median_clean': median_clean,
                        'median_corruption': median_corruption,
                        'difference': difference,
                        'wilcoxon_stat': stat,
                        'p_value': p_value,
                        'significance': significance,
                        'direction': 'corruption_better' if difference > 0 else 'clean_better'
                    }
                    
                    results.append(result)
                    
                    # Log this comparison
                    direction = "↑" if difference > 0 else "↓"
                    sig_mark = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
                    
                    logf.write(f"  {corruption}-trained vs clean-trained:\n")
                    logf.write(f"    Medians: {median_corruption:.4f} vs {median_clean:.4f}\n")
                    logf.write(f"    Difference: {direction} {difference:+.4f}\n")
                    logf.write(f"    Wilcoxon stat: {stat}\n")
                    logf.write(f"    p-value: {p_value:.6f} {sig_mark}\n")
                    logf.write(f"    Significant: {significance}\n")
                    logf.write(f"    Videos: {len(common_videos)}\n")
                    
                except Exception as e:
                    logf.write(f"  ERROR: {str(e)}\n")
                    continue
                
                logf.write("\n")
    
    # Save results to CSV
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_dir / 'baseline_comparison_results.csv', index=False)
        
        # Generate summary table matching user's format
        generate_summary_tables(results_df, output_dir)
        
        print(f"✅ Analysis completed!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"📊 Total comparisons: {len(results)}")
        
        significant_count = results_df['significance'].sum()
        print(f"🔍 Significant differences: {significant_count}/{len(results)} ({significant_count/len(results)*100:.1f}%)")
    
    else:
        print("❌ No valid comparisons found!")
    
    return results

def generate_summary_tables(results_df, output_dir):
    """Generate summary tables matching the user's format"""
    
    metrics = results_df['metric'].unique()
    
    for metric in metrics:
        metric_data = results_df[results_df['metric'] == metric].copy()
        
        # Create summary table
        summary = []
        for _, row in metric_data.iterrows():
            summary.append({
                'CORRUPTION': row['corruption'].replace('_', ' ').title(),
                'DIFFERENCE': row['difference'],
                'WILCOXON_STAT': row['wilcoxon_stat'],
                'P_VALUE': row['p_value'],
                'SIGNIFICANCE': row['significance']
            })
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(output_dir / f'{metric}_summary_table.csv', index=False)
        
        # Generate formatted table for display
        with open(output_dir / f'{metric}_formatted_table.txt', 'w') as f:
            f.write(f"METRIC: {metric.upper()}\n")
            f.write("="*70 + "\n")
            f.write(f"{'CORRUPTION':<20} {'DIFFERENCE':<12} {'WILCOXON_STAT':<15} {'P_VALUE':<12} {'SIGNIFICANCE':<12}\n")
            f.write("-"*70 + "\n")
            
            for _, row in summary_df.iterrows():
                sig_text = "True" if row['SIGNIFICANCE'] else "False"
                f.write(f"{row['CORRUPTION']:<20} {row['DIFFERENCE']:<12.4f} {row['WILCOXON_STAT']:<15} {row['P_VALUE']:<12.6f} {sig_text:<12}\n")
            
            f.write("\nInterpretation:\n")
            f.write("- Difference: Median difference (corruption-trained - clean-trained)\n")
            f.write("- Positive difference: Corruption training improves performance\n")
            f.write("- P < 0.05: Statistically significant difference\n")

def validate_against_user_data(results_df):
    """Cross-validate results against user's expected values"""
    
    print("\n" + "="*50)
    print("VALIDATION AGAINST USER'S EXPECTED VALUES")
    print("="*50)
    
    # Expected values from user's images
    expected_jaccard = {
        'gaussian_noise': {'difference': 0.6227, 'p_value': 0.0256, 'significant': True},
        'motion_blur': {'difference': 18.2148, 'p_value': 0.0000, 'significant': True},
        'defocus_blur': {'difference': 19.8153, 'p_value': 0.0000, 'significant': True},
        'uneven_illumination': {'difference': 17.2548, 'p_value': 0.0000, 'significant': True},
        'smoke_effect': {'difference': 4.5659, 'p_value': 0.2479, 'significant': False},
        'random_corruptions': {'difference': -1.8081, 'p_value': 0.9947, 'significant': False}
    }
    
    jaccard_data = results_df[results_df['metric'] == 'jaccard']
    
    print(f"{'Corruption':<20} {'Expected P':<12} {'Calculated P':<12} {'Match':<8}")
    print("-" * 60)
    
    for corruption, expected in expected_jaccard.items():
        row = jaccard_data[jaccard_data['corruption'] == corruption]
        if len(row) > 0:
            calc_p = row.iloc[0]['p_value']
            expected_p = expected['p_value']
            
            # Check if p-values match within tolerance
            if expected_p == 0.0000:
                match = calc_p < 0.0001
            else:
                match = abs(calc_p - expected_p) < 0.01
            
            match_symbol = "✅" if match else "❌"
            print(f"{corruption:<20} {expected_p:<12.4f} {calc_p:<12.6f} {match_symbol:<8}")
        else:
            print(f"{corruption:<20} {expected['p_value']:<12.4f} {'MISSING':<12} {'❌':<8}")

def main():
    args = parse_args()
    
    # Load and validate data
    df = load_and_validate(args.input)
    
    print("🔍 Starting corruption vs clean baseline analysis...")
    print(f"📊 Input data: {df.shape[0]} rows")
    
    # Run analysis
    results = analyze_corruption_vs_clean(df, args.alpha, args.output)
    
    # Validate against user's expected values
    if results:
        results_df = pd.DataFrame(results)
        validate_against_user_data(results_df)

if __name__ == '__main__':
    main()