#!/usr/bin/env python3
"""
diagnose_pvalue_differences.py

Investigates why calculated p-values don't match expected values.
Tests different statistical approaches to find the correct method.
"""
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu, ttest_rel, ranksums
from pathlib import Path

def load_data():
    df = pd.read_csv('results/all_per_video_metrics.csv')
    return df

def test_different_approaches(df, corruption, metric):
    """Test different statistical approaches for the same comparison"""
    
    print(f"\n=== {corruption.upper()} - {metric.upper()} ===")
    
    # Filter data for this corruption evaluation dataset
    eval_subset = df[df['eval_condition'] == corruption]
    metric_subset = eval_subset[eval_subset['metric'] == metric]
    
    # Pivot to get training conditions as columns
    pivot_data = metric_subset.pivot(
        index='video_id', 
        columns='train_condition', 
        values='score'
    ).dropna()
    
    if 'clean' not in pivot_data.columns or corruption not in pivot_data.columns:
        print("  Missing data!")
        return
    
    clean_values = pivot_data['clean']
    corruption_values = pivot_data[corruption]
    
    # Find common videos
    common_videos = clean_values.index.intersection(corruption_values.index)
    clean_paired = clean_values.loc[common_videos].values
    corruption_paired = corruption_values.loc[common_videos].values
    
    print(f"  Videos: {len(common_videos)}")
    print(f"  Clean median: {np.median(clean_paired):.4f}")
    print(f"  Corruption median: {np.median(corruption_paired):.4f}")
    print(f"  Difference: {np.median(corruption_paired) - np.median(clean_paired):+.4f}")
    
    # Test 1: Wilcoxon signed-rank (paired)
    try:
        stat1, p1 = wilcoxon(clean_paired, corruption_paired)
        print(f"  Wilcoxon signed-rank: p={p1:.6f}, stat={stat1}")
    except Exception as e:
        print(f"  Wilcoxon signed-rank: ERROR - {e}")
    
    # Test 2: Mann-Whitney U (unpaired)
    try:
        stat2, p2 = mannwhitneyu(clean_paired, corruption_paired, alternative='two-sided')
        print(f"  Mann-Whitney U: p={p2:.6f}, stat={stat2}")
    except Exception as e:
        print(f"  Mann-Whitney U: ERROR - {e}")
    
    # Test 3: Paired t-test
    try:
        stat3, p3 = ttest_rel(clean_paired, corruption_paired)
        print(f"  Paired t-test: p={p3:.6f}, stat={stat3}")
    except Exception as e:
        print(f"  Paired t-test: ERROR - {e}")
    
    # Test 4: Wilcoxon rank-sum (unpaired)
    try:
        stat4, p4 = ranksums(clean_paired, corruption_paired)
        print(f"  Wilcoxon rank-sum: p={p4:.6f}, stat={stat4}")
    except Exception as e:
        print(f"  Wilcoxon rank-sum: ERROR - {e}")
    
    # Test 5: Different alternative hypotheses
    try:
        _, p5_greater = wilcoxon(clean_paired, corruption_paired, alternative='greater')
        _, p5_less = wilcoxon(clean_paired, corruption_paired, alternative='less')
        print(f"  Wilcoxon (greater): p={p5_greater:.6f}")
        print(f"  Wilcoxon (less): p={p5_less:.6f}")
    except Exception as e:
        print(f"  Wilcoxon alternatives: ERROR - {e}")

def check_data_integrity(df):
    """Check for data quality issues that might affect p-values"""
    
    print("=== DATA INTEGRITY CHECK ===")
    
    # Check for missing values
    missing = df.isnull().sum()
    print(f"Missing values: {missing.sum()}")
    
    # Check for duplicate entries
    duplicates = df.duplicated(['video_id', 'train_condition', 'eval_condition', 'metric']).sum()
    print(f"Duplicate entries: {duplicates}")
    
    # Check video coverage
    print("\nVideo coverage by condition:")
    for condition in sorted(df['train_condition'].unique()):
        videos = df[df['train_condition'] == condition]['video_id'].nunique()
        print(f"  {condition}: {videos} videos")
    
    # Check for outliers
    print("\nScore distributions:")
    for metric in sorted(df['metric'].unique()):
        metric_data = df[df['metric'] == metric]['score']
        print(f"  {metric}: min={metric_data.min():.2f}, max={metric_data.max():.2f}, mean={metric_data.mean():.2f}")

def validate_against_expected():
    """Compare against expected values from user's tables"""
    
    expected_jaccard = {
        'gaussian_noise': {'difference': 0.6227, 'p_value': 0.0256, 'significant': True},
        'motion_blur': {'difference': 18.2148, 'p_value': 0.0000, 'significant': True},
        'defocus_blur': {'difference': 19.8153, 'p_value': 0.0000, 'significant': True},
        'uneven_illumination': {'difference': 17.2548, 'p_value': 0.0000, 'significant': True},
        'smoke_effect': {'difference': 4.5659, 'p_value': 0.2479, 'significant': False},
        'random_corruptions': {'difference': -1.8081, 'p_value': 0.9947, 'significant': False}
    }
    
    print("\n=== EXPECTED VALUES COMPARISON ===")
    print("Note: These are the values from your tables that we're trying to match")
    
    for corruption, expected in expected_jaccard.items():
        print(f"\n{corruption.upper()}: Expected diff={expected['difference']:.4f}, p={expected['p_value']:.4f}")

def main():
    df = load_data()
    
    check_data_integrity(df)
    validate_against_expected()
    
    # Test problematic cases
    problem_cases = [
        ('gaussian_noise', 'jaccard'),
        ('smoke_effect', 'jaccard'),
        ('motion_blur', 'jaccard'),  # This one should match
        ('defocus_blur', 'jaccard')  # This one should match
    ]
    
    print("\n" + "="*50)
    print("TESTING DIFFERENT STATISTICAL APPROACHES")
    print("="*50)
    
    for corruption, metric in problem_cases:
        test_different_approaches(df, corruption, metric)
    
    print("\n" + "="*50)
    print("RECOMMENDATIONS")
    print("="*50)
    print("1. Check if your original analysis used a different statistical test")
    print("2. Verify data preprocessing steps (filtering, outlier removal)")
    print("3. Confirm video matching between conditions")
    print("4. Consider if one-tailed vs two-tailed tests were used")
    print("5. Check if any data transformations were applied")

if __name__ == '__main__':
    main()