#!/usr/bin/env python3
"""
alternative_analysis_approaches.py

Tests different interpretation of what the user's tables might represent.
"""
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from pathlib import Path

def load_data():
    df = pd.read_csv('results/all_per_video_metrics.csv')
    return df

def approach_1_clean_on_corruption_vs_corruption_on_corruption(df):
    """
    Test if the comparison is: clean-trained evaluated on corruption vs corruption-trained evaluated on corruption
    This would explain the larger effect sizes.
    """
    print("\n=== APPROACH 1: Clean-on-Corruption vs Corruption-on-Corruption ===")
    
    expected_jaccard = {
        'gaussian_noise': {'difference': 0.6227, 'p_value': 0.0256},
        'motion_blur': {'difference': 18.2148, 'p_value': 0.0000},
        'defocus_blur': {'difference': 19.8153, 'p_value': 0.0000},
        'uneven_illumination': {'difference': 17.2548, 'p_value': 0.0000},
        'smoke_effect': {'difference': 4.5659, 'p_value': 0.2479},
        'random_corruptions': {'difference': -1.8081, 'p_value': 0.9947}
    }
    
    for corruption, expected in expected_jaccard.items():
        print(f"\n{corruption.upper()}:")
        
        # Get clean-trained model evaluated on corruption dataset
        clean_on_corruption = df[
            (df['train_condition'] == 'clean') & 
            (df['eval_condition'] == corruption) & 
            (df['metric'] == 'jaccard')
        ]
        
        # Get corruption-trained model evaluated on corruption dataset  
        corruption_on_corruption = df[
            (df['train_condition'] == corruption) & 
            (df['eval_condition'] == corruption) & 
            (df['metric'] == 'jaccard')
        ]
        
        if len(clean_on_corruption) == 0 or len(corruption_on_corruption) == 0:
            print(f"  Missing data!")
            continue
            
        # Merge on video_id to get paired comparisons
        merged = pd.merge(
            clean_on_corruption[['video_id', 'score']].rename(columns={'score': 'clean_score'}),
            corruption_on_corruption[['video_id', 'score']].rename(columns={'score': 'corruption_score'}),
            on='video_id'
        )
        
        if len(merged) < 5:
            print(f"  Insufficient paired data: {len(merged)} videos")
            continue
            
        clean_scores = merged['clean_score'].values
        corruption_scores = merged['corruption_score'].values
        
        median_clean = np.median(clean_scores)
        median_corruption = np.median(corruption_scores)
        difference = median_corruption - median_clean
        
        try:
            stat, p_value = wilcoxon(clean_scores, corruption_scores)
            
            print(f"  Videos: {len(merged)}")
            print(f"  Clean-on-{corruption} median: {median_clean:.4f}")
            print(f"  {corruption}-on-{corruption} median: {median_corruption:.4f}")
            print(f"  Difference: {difference:+.4f} (expected: {expected['difference']:+.4f})")
            print(f"  P-value: {p_value:.6f} (expected: {expected['p_value']:.4f})")
            
            # Check if this matches better
            diff_match = abs(difference - expected['difference']) < 1.0
            if expected['p_value'] == 0.0000:
                p_match = p_value < 0.0001
            else:
                p_match = abs(p_value - expected['p_value']) < 0.05
                
            match_symbol = "✅" if (diff_match and p_match) else "❌"
            print(f"  Match: {match_symbol}")
            
        except Exception as e:
            print(f"  ERROR: {e}")

def approach_2_percentage_differences(df):
    """
    Test if differences are calculated as percentages rather than absolute values
    """
    print("\n=== APPROACH 2: Percentage Differences ===")
    
    corruptions = ['gaussian_noise', 'motion_blur', 'defocus_blur', 'uneven_illumination', 'smoke_effect', 'random_corruptions']
    
    for corruption in corruptions:
        # Same data as before but calculate percentage difference
        eval_subset = df[df['eval_condition'] == corruption]
        metric_subset = eval_subset[eval_subset['metric'] == 'jaccard']
        
        pivot_data = metric_subset.pivot(
            index='video_id', 
            columns='train_condition', 
            values='score'
        ).dropna()
        
        if 'clean' not in pivot_data.columns or corruption not in pivot_data.columns:
            continue
            
        clean_values = pivot_data['clean']
        corruption_values = pivot_data[corruption]
        
        common_videos = clean_values.index.intersection(corruption_values.index)
        clean_paired = clean_values.loc[common_videos].values
        corruption_paired = corruption_values.loc[common_videos].values
        
        median_clean = np.median(clean_paired)
        median_corruption = np.median(corruption_paired)
        
        # Calculate percentage difference
        abs_diff = median_corruption - median_clean
        pct_diff = (abs_diff / median_clean) * 100 if median_clean != 0 else 0
        
        print(f"{corruption}: abs={abs_diff:+.4f}, pct={pct_diff:+.2f}%")

def approach_3_different_baseline_models(df):
    """
    Check if there are other baseline models or evaluation setups
    """
    print("\n=== APPROACH 3: All Available Comparisons ===")
    
    print("Available train-eval combinations:")
    combinations = df.groupby(['train_condition', 'eval_condition']).size().reset_index(name='count')
    for _, row in combinations.iterrows():
        print(f"  Train: {row['train_condition']} → Eval: {row['eval_condition']} ({row['count']} entries)")

def main():
    df = load_data()
    
    print("Testing alternative interpretations of your analysis...")
    
    approach_1_clean_on_corruption_vs_corruption_on_corruption(df)
    approach_2_percentage_differences(df)
    approach_3_different_baseline_models(df)
    
    print("\n" + "="*60)
    print("CONCLUSIONS")
    print("="*60)
    print("Based on the analysis, your original tables likely represent:")
    print("1. Clean-trained model evaluated on corruption datasets")
    print("   vs Corruption-trained model evaluated on same corruption datasets")
    print("2. This explains the larger effect sizes in your expected values")
    print("3. The p-value differences suggest possible data filtering or")
    print("   different statistical test implementations")

if __name__ == '__main__':
    main()