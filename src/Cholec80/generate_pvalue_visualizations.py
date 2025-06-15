#!/usr/bin/env python3
"""
generate_pvalue_visualizations.py

Creates multiple visualizations for p-values from statistical analysis:
1. Heatmap/Confusion Matrix of p-values (Corruption vs Metric)
2. Significance matrix (binary: significant/not significant)
3. Effect size heatmap (difference values)
4. Combined visualization with annotations

Input: statistical_log.txt file from analyze_test.py
Output: Multiple PNG files with different visualizations
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def parse_statistical_log(file_path):
    """
    Parse the statistical_log.txt file and extract data into a structured format.
    
    Returns:
        pandas.DataFrame with columns: corruption, metric, p_value, significance, difference, etc.
    """
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract alpha level
    alpha_match = re.search(r'Alpha = ([\d.]+)', content)
    alpha = float(alpha_match.group(1)) if alpha_match else 0.05
    
    # Split by corruption sections
    corruption_sections = re.split(r'=== Corruption: (.+?) ===', content)[1:]  # Skip first empty part
    
    data_rows = []
    
    # Process pairs: (corruption_name, section_content)
    for i in range(0, len(corruption_sections), 2):
        corruption = corruption_sections[i].strip()
        section_content = corruption_sections[i + 1]
        
        # Extract data lines (skip header)
        lines = section_content.strip().split('\n')
        
        # Find the header line to understand the structure
        header_line = None
        data_lines = []
        
        for line in lines:
            if 'metric' in line and 'p-value' in line:
                header_line = line
            elif line.strip() and not line.startswith('#') and 'metric' not in line:
                data_lines.append(line)
        
        # Parse each data line
        for line in data_lines:
            # Use regex to extract fields (handles varying whitespace)
            parts = line.split()
            if len(parts) >= 8:  # Ensure we have enough parts
                metric = parts[0]
                pairs = int(parts[1])
                median = float(parts[2])
                # Skip train_condition and eval_condition (can be multi-word)
                difference = float(parts[-4])  # 4th from end
                wilcoxon_stat = float(parts[-3])  # 3rd from end  
                p_value = float(parts[-2])  # 2nd from end
                significance = parts[-1] == 'True'  # Last element
                
                data_rows.append({
                    'corruption': corruption,
                    'metric': metric,
                    'pairs': pairs,
                    'median': median,
                    'difference': difference,
                    'wilcoxon_stat': wilcoxon_stat,
                    'p_value': p_value,
                    'significance': significance,
                    'alpha': alpha
                })
    
    df = pd.DataFrame(data_rows)
    return df

def create_pvalue_heatmap(df, output_path):
    """Create a heatmap of p-values (Corruption vs Metric)"""
    
    # Pivot the data to create matrix format
    pvalue_matrix = df.pivot(index='corruption', columns='metric', values='p_value')
    
    # Create the heatmap
    plt.figure(figsize=(10, 8))
    
    # Use a diverging colormap where low p-values (significant) are dark
    ax = sns.heatmap(
        pvalue_matrix, 
        annot=True, 
        fmt='.4f',
        cmap='RdYlBu_r',  # Red for low p-values (significant), Blue for high
        center=0.05,  # Center the colormap at alpha=0.05
        cbar_kws={'label': 'p-value'},
        linewidths=0.5,
        square=True
    )
    
    plt.title('P-Values Heatmap\n(Darker = More Significant)', fontsize=16, fontweight='bold')
    plt.xlabel('Metrics', fontsize=12, fontweight='bold')
    plt.ylabel('Corruption Types', fontsize=12, fontweight='bold')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add a horizontal line at alpha=0.05
    plt.axhline(y=0, color='black', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ P-value heatmap saved to: {output_path}")

def create_significance_matrix(df, output_path):
    """Create a binary significance matrix"""
    
    # Pivot the data for significance
    sig_matrix = df.pivot(index='corruption', columns='metric', values='significance')
    
    # Convert boolean to int for better visualization
    sig_matrix = sig_matrix.astype(int)
    
    plt.figure(figsize=(10, 8))
    
    # Create custom colormap: 0=white (not significant), 1=dark red (significant)
    colors = ['white', 'darkred']
    n_bins = 2
    cmap = plt.cm.colors.ListedColormap(colors)
    
    ax = sns.heatmap(
        sig_matrix,
        annot=True,
        fmt='d',
        cmap=cmap,
        cbar_kws={'label': 'Significant (1) / Not Significant (0)'},
        linewidths=1,
        square=True,
        vmin=0,
        vmax=1
    )
    
    plt.title('Statistical Significance Matrix\n(Red = Significant, White = Not Significant)', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Metrics', fontsize=12, fontweight='bold')
    plt.ylabel('Corruption Types', fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Significance matrix saved to: {output_path}")

def create_effect_size_heatmap(df, output_path):
    """Create a heatmap of effect sizes (difference values)"""
    
    # Pivot the data for differences
    diff_matrix = df.pivot(index='corruption', columns='metric', values='difference')
    
    plt.figure(figsize=(10, 8))
    
    # Use a diverging colormap centered at 0
    ax = sns.heatmap(
        diff_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',  # Red for positive (improvement), Blue for negative
        center=0,
        cbar_kws={'label': 'Effect Size (Difference from Baseline)'},
        linewidths=0.5,
        square=True
    )
    
    plt.title('Effect Size Heatmap\n(Red = Improvement, Blue = Degradation)', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Metrics', fontsize=12, fontweight='bold')
    plt.ylabel('Corruption Types', fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Effect size heatmap saved to: {output_path}")

def create_combined_visualization(df, output_path):
    """Create a combined visualization with p-values and significance markers"""
    
    # Prepare matrices
    pvalue_matrix = df.pivot(index='corruption', columns='metric', values='p_value')
    sig_matrix = df.pivot(index='corruption', columns='metric', values='significance')
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Left subplot: P-values with significance annotations
    sns.heatmap(
        pvalue_matrix,
        annot=True,
        fmt='.4f',
        cmap='RdYlBu_r',
        center=0.05,
        ax=ax1,
        cbar_kws={'label': 'p-value'},
        linewidths=0.5,
        square=True
    )
    
    # Add asterisks for significant results
    for i, corruption in enumerate(pvalue_matrix.index):
        for j, metric in enumerate(pvalue_matrix.columns):
            if sig_matrix.loc[corruption, metric]:
                ax1.text(j + 0.5, i + 0.8, '*', fontsize=20, fontweight='bold', 
                        ha='center', va='center', color='black')
    
    ax1.set_title('P-Values with Significance Markers\n(* = Significant at α=0.05)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Corruption Types', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    
    # Right subplot: Effect sizes
    diff_matrix = df.pivot(index='corruption', columns='metric', values='difference')
    sns.heatmap(
        diff_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        ax=ax2,
        cbar_kws={'label': 'Effect Size'},
        linewidths=0.5,
        square=True
    )
    
    ax2.set_title('Effect Sizes\n(Red = Improvement, Blue = Degradation)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Corruption Types', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Combined visualization saved to: {output_path}")

def generate_summary_stats(df, output_path):
    """Generate and save summary statistics"""
    
    with open(output_path, 'w') as f:
        f.write("=== STATISTICAL ANALYSIS SUMMARY ===\n\n")
        
        alpha = df['alpha'].iloc[0]
        f.write(f"Significance Level (α): {alpha}\n")
        f.write(f"Total Comparisons: {len(df)}\n")
        f.write(f"Significant Results: {df['significance'].sum()}\n")
        f.write(f"Non-Significant Results: {(~df['significance']).sum()}\n")
        f.write(f"Significance Rate: {df['significance'].mean():.1%}\n\n")
        
        f.write("=== BY CORRUPTION TYPE ===\n")
        corruption_stats = df.groupby('corruption').agg({
            'significance': ['count', 'sum', 'mean'],
            'p_value': ['min', 'max', 'mean'],
            'difference': ['mean', 'std']
        }).round(4)
        f.write(corruption_stats.to_string())
        f.write("\n\n")
        
        f.write("=== BY METRIC ===\n")
        metric_stats = df.groupby('metric').agg({
            'significance': ['count', 'sum', 'mean'],
            'p_value': ['min', 'max', 'mean'],
            'difference': ['mean', 'std']
        }).round(4)
        f.write(metric_stats.to_string())
        f.write("\n\n")
        
        f.write("=== MOST SIGNIFICANT RESULTS (p < 0.01) ===\n")
        highly_sig = df[df['p_value'] < 0.01].sort_values('p_value')
        for _, row in highly_sig.iterrows():
            f.write(f"{row['corruption']} - {row['metric']}: p={row['p_value']:.6f}, diff={row['difference']:.2f}\n")
    
    print(f"✅ Summary statistics saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate p-value visualizations from statistical analysis")
    parser.add_argument('--input', 
                       default='/home/santhi/Documents/DACAT/src/Cholec80/results/statistical_log.txt',
                       help='Path to statistical_log.txt file')
    parser.add_argument('--output_dir', 
                       default='/home/santhi/Documents/DACAT/src/Cholec80/results/visualizations',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse the statistical log file
    print(f"📖 Parsing statistical log: {args.input}")
    df = parse_statistical_log(args.input)
    
    print(f"📊 Found {len(df)} statistical comparisons:")
    print(f"   - Corruptions: {df['corruption'].unique()}")
    print(f"   - Metrics: {df['metric'].unique()}")
    print(f"   - Significant results: {df['significance'].sum()}/{len(df)}")
    
    # Generate visualizations
    print("\n🎨 Generating visualizations...")
    
    create_pvalue_heatmap(df, output_dir / 'pvalue_heatmap.png')
    create_significance_matrix(df, output_dir / 'significance_matrix.png') 
    create_effect_size_heatmap(df, output_dir / 'effect_size_heatmap.png')
    create_combined_visualization(df, output_dir / 'combined_analysis.png')
    
    # Generate summary statistics
    generate_summary_stats(df, output_dir / 'summary_statistics.txt')
    
    print(f"\n🎉 All visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    print("  📈 pvalue_heatmap.png - P-values color-coded by significance")
    print("  🔴 significance_matrix.png - Binary significance matrix")
    print("  📊 effect_size_heatmap.png - Effect sizes (improvements/degradations)")
    print("  📋 combined_analysis.png - Combined p-values and effect sizes")
    print("  📄 summary_statistics.txt - Numerical summary")

if __name__ == '__main__':
    main()
