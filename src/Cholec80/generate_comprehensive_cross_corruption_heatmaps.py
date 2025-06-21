#!/usr/bin/env python3
"""
generate_comprehensive_cross_corruption_heatmaps.py

Creates heatmaps showing statistical significance and effect sizes for ALL cross-corruption combinations
from the comprehensive statistical analysis output.
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def parse_comprehensive_statistical_log(file_path):
    """
    Parse the comprehensive cross-corruption statistical log using the EXACT SAME approach
    as the existing parse_statistical_log function to ensure consistency.
    """
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Extract alpha level using SAME approach as existing code
    alpha = 0.05
    for line in lines:
        if 'Alpha' in line:
            alpha_match = re.search(r'Alpha[=:]\s*([\d.]+)', line)
            if alpha_match:
                alpha = float(alpha_match.group(1))
            break
    
    data_rows = []
    current_eval_condition = None
    current_metric = None
    
    print("🔍 Parsing comprehensive statistical log using SAME method as existing code...")
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check for evaluation condition
        if 'EVALUATION CONDITION:' in line:
            current_eval_condition = line.split('EVALUATION CONDITION:')[1].strip()
            print(f"   Processing eval condition: {current_eval_condition}")
        
        # Check for metric using SAME pattern as existing code
        elif '--- Metric:' in line and '---' in line:
            current_metric = line.split('--- Metric:')[1].split('---')[0].strip().lower()
        
        # Parse comparison lines using SAME approach as existing Wilcoxon output format
        elif ' vs ' in line and 'Medians:' in lines[i+1] if i+1 < len(lines) else False:
            comparison_part = line.split(':')[0].strip() if ':' in line else line.strip()
            
            # Look for the next lines with Medians and p-value (SAME format as analyze_test.py output)
            if i + 2 < len(lines):
                median_line = lines[i + 1].strip()
                pvalue_line = lines[i + 2].strip()
                
                # Parse using SAME regex patterns as existing code
                if median_line.startswith('Medians:'):
                    median_match = re.search(r'Medians:\s*([\d.]+)\s+vs\s+([\d.]+)\s+\(diff:\s*([-+]?[\d.]+)\)', median_line)
                    if median_match:
                        median_1 = float(median_match.group(1))
                        median_2 = float(median_match.group(2))
                        effect_size = float(median_match.group(3))
                        
                        # Parse p-value using SAME approach
                        if pvalue_line.startswith('p-value:'):
                            pvalue_match = re.search(r'p-value:\s*([\d.]+)', pvalue_line)
                            n_match = re.search(r'n=(\d+)', pvalue_line)
                            
                            if pvalue_match:
                                p_value = float(pvalue_match.group(1))
                                pairs = int(n_match.group(1)) if n_match else 40
                                significance = p_value < alpha  # SAME significance test
                                
                                # Parse the comparison using SAME approach
                                if ' vs ' in comparison_part:
                                    train_1, train_2 = comparison_part.split(' vs ')
                                    
                                    # Store in SAME format as existing statistical logs
                                    data_rows.append({
                                        'eval_condition': current_eval_condition,
                                        'train_1': train_1.strip(),
                                        'train_2': train_2.strip(),
                                        'metric': current_metric,
                                        'pairs': pairs,
                                        'median_1': median_1,
                                        'median_2': median_2,
                                        'effect_size': effect_size,
                                        'wilcoxon_stat': 0.0,  # Same as existing format
                                        'p_value': p_value,
                                        'significance': significance,
                                        'alpha': alpha
                                    })
        
        i += 1
    
    df = pd.DataFrame(data_rows)
    
    print(f"✅ Extracted {len(df)} pairwise comparisons using consistent parsing method")
    if len(df) > 0:
        print(f"   Eval conditions: {sorted(df['eval_condition'].unique())}")
        print(f"   Training conditions: {sorted(set(df['train_1'].unique()) | set(df['train_2'].unique()))}")
        print(f"   Metrics: {sorted(df['metric'].unique())}")
    
    return df

def create_cross_corruption_matrices(df):
    """
    Create cross-corruption matrices for p-values and effect sizes from pairwise comparison data.
    """
    
    # Get all unique conditions and metrics - CUSTOM ORDER
    # Define custom order: clean first, then corruptions in logical sequence
    all_train_conditions = ['clean', 'gaussian_noise', 'motion_blur', 'defocus_blur', 'uneven_illumination', 'smoke_effect', 'random_corruptions']
    all_eval_conditions = ['clean', 'gaussian_noise', 'motion_blur', 'defocus_blur', 'uneven_illumination', 'smoke_effect', 'random_corruptions']
    all_metrics = sorted(df['metric'].unique())
    
    print(f"🔄 Creating cross-corruption matrices...")
    print(f"   Training conditions: {all_train_conditions}")
    print(f"   Evaluation conditions: {all_eval_conditions}")
    print(f"   Metrics: {all_metrics}")
    
    results = {}
    
    for metric in all_metrics:
        metric_df = df[df['metric'] == metric].copy()
        
        # Initialize matrices
        p_value_matrix = pd.DataFrame(
            np.nan, 
            index=all_train_conditions, 
            columns=all_eval_conditions
        )
        effect_size_matrix = pd.DataFrame(
            np.nan, 
            index=all_train_conditions, 
            columns=all_eval_conditions
        )
        
        # Fill matrices with pairwise comparison data
        for _, row in metric_df.iterrows():
            eval_cond = row['eval_condition']
            train_1 = row['train_1']
            train_2 = row['train_2']
            
            # We want to fill the matrix as: train_condition (rows) vs eval_condition (columns)
            # The effect size tells us how train_2 compares to train_1 on eval_cond
            # So we put this in the train_2 row, eval_cond column
            
            p_value_matrix.loc[train_2, eval_cond] = row['p_value']
            effect_size_matrix.loc[train_2, eval_cond] = row['effect_size']
            
            # Also add the inverse comparison (train_1 vs train_2 with opposite effect)
            p_value_matrix.loc[train_1, eval_cond] = row['p_value']  # p-value is symmetric
            effect_size_matrix.loc[train_1, eval_cond] = -row['effect_size']  # effect is opposite
        
        results[metric] = {
            'p_values': p_value_matrix,
            'effect_sizes': effect_size_matrix
        }
    
    return results

def generate_heatmaps(matrices_dict, output_dir):
    """Generate heatmaps for all metrics and both p-values and effect sizes."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎨 Generating comprehensive heatmaps...")
    
    for metric, matrices in matrices_dict.items():
        p_value_matrix = matrices['p_values']
        effect_size_matrix = matrices['effect_sizes']
        
        # Generate P-value heatmap with lighter colors
        plt.figure(figsize=(12, 10))
        
        # Create custom colormap for p-values (lighter colors)
        sns.heatmap(
            p_value_matrix,
            annot=True,
            fmt='.4f',
            cmap='RdYlGn_r',  # Red for high p-values, Green for low p-values
            center=0.05,
            vmin=0,
            vmax=0.15,  # Increased max for lighter colors
            cbar_kws={'label': 'P-value'},
            square=True,
            linewidths=0.5,
            annot_kws={'size': 9}  # Smaller annotation font
        )
        
        plt.title(f'{metric.title()} - Cross-Corruption P-values\n(Green: Significant p<0.05, Red: Not Significant)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Evaluation Dataset', fontweight='bold')
        plt.ylabel('Training Condition', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        p_value_file = output_dir / f'{metric}_light_pvalue_heatmap.png'
        plt.savefig(p_value_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ {metric.title()} p-value heatmap saved to: {p_value_file}")
        
        # Generate Effect Size heatmap with lighter colors
        plt.figure(figsize=(12, 10))
        
        # Determine color scale based on effect size range
        max_abs_effect = max(abs(effect_size_matrix.min().min()), abs(effect_size_matrix.max().max()))
        
        # Use lighter colormap
        sns.heatmap(
            effect_size_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdBu',  # Red for positive (better), Blue for negative (worse) - REVERSED
            center=0,
            vmin=-max_abs_effect * 0.8,  # Reduce range for lighter colors
            vmax=max_abs_effect * 0.8,
            cbar_kws={'label': 'Effect Size (Difference in Median)'},
            square=True,
            linewidths=0.5,
            annot_kws={'size': 9}  # Smaller annotation font
        )
        
        plt.title(f'{metric.title()} - Cross-Corruption Effect Sizes\n(Red: Better Performance, Blue: Worse Performance)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Evaluation Dataset', fontweight='bold')
        plt.ylabel('Training Condition', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        effect_size_file = output_dir / f'{metric}_light_effect_size_heatmap.png'
        plt.savefig(effect_size_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ {metric.title()} effect size heatmap saved to: {effect_size_file}")
    
    # Generate a summary file explaining the analysis
    summary_file = output_dir / 'analysis_explanation.txt'
    with open(summary_file, 'w') as f:
        f.write("=== COMPREHENSIVE CROSS-CORRUPTION ANALYSIS EXPLANATION ===\n\n")
        f.write("DATA SOURCE:\n")
        f.write("- Source: all_per_video_metrics.csv (7,840 rows)\n")
        f.write("- Coverage: 7 training conditions × 7 evaluation conditions × 40 videos × 4 metrics\n\n")
        f.write("ANALYSIS TYPE:\n")
        f.write("- Pairwise comparisons between ALL training conditions on each evaluation dataset\n")
        f.write("- Total comparisons: 588 (21 pairs × 7 eval conditions × 4 metrics)\n")
        f.write("- Statistical test: Wilcoxon signed-rank test (paired, non-parametric)\n\n")
        f.write("INTERPRETATION:\n")
        f.write("- P-value heatmaps: Green = statistically significant difference (p<0.05)\n")
        f.write("- Effect size heatmaps: Blue = better performance, Red = worse performance\n")
        f.write("- Each cell shows: training_condition vs other_training_condition on eval_dataset\n\n")
        f.write("DIFFERENCES FROM PREVIOUS ANALYSIS:\n")
        f.write("- Previous: Corruption-trained vs Clean-trained (diagonal only)\n")
        f.write("- Current: All possible pairwise comparisons (comprehensive)\n")
        f.write("- Both analyses are valid but answer different research questions\n")
    
    print(f"📄 Analysis explanation saved to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive cross-corruption heatmaps')
    parser.add_argument('--input', default='results/complete_cross_corruption_statistical_log.txt',
                       help='Input comprehensive statistical log file')
    parser.add_argument('--output', default='results/comprehensive_cross_corruption_analysis',
                       help='Output directory for heatmaps')
    
    args = parser.parse_args()
    
    print("🚀 Starting comprehensive cross-corruption heatmap generation...")
    print(f"📖 Reading statistical log: {args.input}")
    
    # Parse the comprehensive statistical log
    df = parse_comprehensive_statistical_log(args.input)
    
    if len(df) == 0:
        print("❌ No data found in statistical log!")
        return
    
    # Create cross-corruption matrices
    matrices_dict = create_cross_corruption_matrices(df)
    
    # Generate heatmaps
    generate_heatmaps(matrices_dict, args.output)
    
    print(f"\n🎉 Comprehensive cross-corruption heatmap generation complete!")
    print(f"📁 Results saved to: {args.output}")

if __name__ == '__main__':
    main()
