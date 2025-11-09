#!/usr/bin/env python3
"""
generate_cross_corruption_heatmaps.py

Creates heatmaps showing how corruption-trained models perform on different datasets.
This implements the cross-corruption analysis discussed with your guide:

X-axis: Corruption-trained models [gaussian_noise, motion_blur, defocus_blur, uneven_illumination, smoke_effect, random_corruption]
Y-axis: Evaluation datasets [clean, gaussian_noise, motion_blur, defocus_blur, uneven_illumination, smoke_effect, random_corruption]

Color coding:
- Green: Positive effect (better performance)
- Red: Negative effect (worse performance)
- White/Yellow: Neutral effect

Generates separate heatmaps for each metric (accuracy, jaccard, precision, recall).
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def parse_statistical_log_cross_corruption(file_path):
    """
    Parse statistical_log.txt and extract cross-corruption comparison data.
    
    Note: Current data only contains same-corruption comparisons (train=eval).
    This function will extract available data and indicate missing combinations.
    
    Returns:
        pandas.DataFrame with columns: train_corruption, eval_corruption, metric, p_value, effect_size, etc.
    """
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract alpha level
    alpha_match = re.search(r'Alpha[=:]\s*([\d.]+)', content)
    alpha = float(alpha_match.group(1)) if alpha_match else 0.05
    
    # Split by corruption sections
    corruption_sections = re.split(r'=== Corruption: (.+?) ===', content)[1:]
    
    data_rows = []
    
    print("🔍 Parsing statistical log...")
    print("📊 Current data format: Same-corruption comparisons only (train=eval)")
    
    # Process pairs: (corruption_name, section_content)
    for i in range(0, len(corruption_sections), 2):
        eval_corruption = corruption_sections[i].strip()
        section_content = corruption_sections[i + 1]
        
        print(f"   Processing: {eval_corruption}")
        
        # Extract data lines
        lines = section_content.strip().split('\n')
        data_lines = [line for line in lines if line.strip() and not line.startswith('#') and 'metric' not in line]
        
        # Parse each data line
        for line in data_lines:
            parts = line.split()
            if len(parts) >= 8:
                metric = parts[0]
                pairs = int(parts[1])
                median = float(parts[2])
                
                # For current data format, train_condition = eval_condition
                # The comparison is: corruption_trained vs clean_trained (both evaluated on corruption)
                train_corruption = eval_corruption  # This is the corruption-trained model
                
                difference = float(parts[-4])  # 4th from end
                wilcoxon_stat = float(parts[-3])  # 3rd from end  
                p_value = float(parts[-2])  # 2nd from end
                significance = parts[-1] == 'True'  # Last element
                
                data_rows.append({
                    'train_corruption': train_corruption,
                    'eval_corruption': eval_corruption,
                    'metric': metric,
                    'pairs': pairs,
                    'median': median,
                    'effect_size': difference,  # Rename for clarity
                    'wilcoxon_stat': wilcoxon_stat,
                    'p_value': p_value,
                    'significance': significance,
                    'alpha': alpha,
                    'comparison_type': 'corruption_vs_clean'  # Mark the type of comparison
                })
    
    df = pd.DataFrame(data_rows)
    
    print(f"✅ Extracted {len(df)} comparisons from statistical log")
    print(f"   Corruption types: {df['train_corruption'].unique()}")
    print(f"   Metrics: {df['metric'].unique()}")
    
    return df

def create_cross_corruption_matrix_data(df):
    """
    Create a cross-corruption matrix from available data.
    
    Current limitation: We only have diagonal comparisons (train=eval corruption).
    This function will:
    1. Place available data on the diagonal
    2. Exclude invalid clean vs clean combinations 
    3. Mark missing cross-corruption combinations
    4. Use the EXACT order specified by user: Gaussian_noise, Motion_blur, Defocus_blur, Uneven_illumination, Smoke_effect, Random_corruption
    """
    
    # Define the EXACT order requested by user
    # User-specified order for corruptions (matching your experimental setup)
    user_specified_order = [
        'gaussian_noise',      # Gaussian_noise
        'motion_blur',         # Motion_blur
        'defocus_blur',        # Defocus_blur
        'smoke_effect',        # Smoke_effect
        'uneven_illumination', # Uneven_illumination
        'random_corruptions'   # Random_corruption (50:50)
    ]    # Get available corruptions from data
    available_corruptions = list(df['train_corruption'].unique())
    
    # Create ordered list: only include corruptions that exist in your data, in the specified order
    ordered_corruptions = []
    for specified_corr in user_specified_order:
        if specified_corr in available_corruptions:
            ordered_corruptions.append(specified_corr)
    
    # Add any remaining corruptions not in the specified order (as fallback)
    for corr in available_corruptions:
        if corr not in ordered_corruptions:
            ordered_corruptions.append(corr)
    
    # Add 'clean' at the beginning as baseline
    all_corruptions = ['clean'] + ordered_corruptions
    
    # Get all metrics
    all_metrics = sorted(df['metric'].unique())
    
    print(f"🔄 Creating cross-corruption matrix with USER SPECIFIED order...")
    print(f"   User specified order: {user_specified_order}")
    print(f"   Available corruptions: {available_corruptions}")
    print(f"   Final ordered corruptions: {ordered_corruptions}")
    print(f"   Full matrix order: {all_corruptions}")
    print(f"   Full matrix size: {len(all_corruptions)} x {len(all_corruptions)}")
    print(f"   Note: Only diagonal elements have real data, clean-clean excluded")
    
    # Create full matrix structure
    full_data = []
    
    for metric in all_metrics:
        metric_df = df[df['metric'] == metric].copy()
        
        for eval_corruption in all_corruptions:
            for train_corruption in all_corruptions:
                
                # EXCLUDE the invalid clean vs clean combination
                if train_corruption == 'clean' and eval_corruption == 'clean':
                    continue  # Skip this invalid combination
                
                # Check if we have real data for this combination
                real_data = metric_df[
                    (metric_df['eval_corruption'] == eval_corruption) & 
                    (metric_df['train_corruption'] == train_corruption)
                ]
                
                if len(real_data) > 0:
                    # Use real data
                    row = real_data.iloc[0]
                    full_data.append({
                        'train_corruption': train_corruption,
                        'eval_corruption': eval_corruption,
                        'metric': metric,
                        'effect_size': row['effect_size'],
                        'p_value': row['p_value'],
                        'significance': row['significance'],
                        'data_type': 'real'
                    })
                else:
                    # Missing cross-corruption data
                    full_data.append({
                        'train_corruption': train_corruption,
                        'eval_corruption': eval_corruption,
                        'metric': metric,
                        'effect_size': np.nan,
                        'p_value': np.nan,
                        'significance': False,
                        'data_type': 'missing'
                    })
    
    full_df = pd.DataFrame(full_data)
    
    # Count data availability
    real_data_count = (full_df['data_type'] == 'real').sum()
    missing_data_count = (full_df['data_type'] == 'missing').sum()
    
    print(f"   Real data points: {real_data_count}")
    print(f"   Missing data points: {missing_data_count}")
    print(f"   Clean-clean combination excluded (invalid)")
    print(f"   Data availability: {real_data_count/(real_data_count+missing_data_count)*100:.1f}%")
    
    return full_df, all_corruptions, all_metrics

def create_metric_heatmap(df, metric, all_corruptions, output_path):
    """Create p-value heatmap for a specific metric"""
    
    # Filter data for this metric
    metric_data = df[df['metric'] == metric].copy()
    
    # Create pivot table for p-values
    pvalue_matrix = metric_data.pivot_table(
        index='eval_corruption',
        columns='train_corruption', 
        values='p_value',
        fill_value=np.nan
    )
    
    # Reorder according to specified order
    ordered_corruptions = [c for c in all_corruptions if c in pvalue_matrix.index]
    pvalue_matrix = pvalue_matrix.reindex(index=ordered_corruptions, columns=ordered_corruptions)
    
    # Create effect size matrix for annotations (to show direction)
    effect_matrix = metric_data.pivot_table(
        index='eval_corruption',
        columns='train_corruption',
        values='effect_size',
        fill_value=np.nan
    ).reindex(index=ordered_corruptions, columns=ordered_corruptions)
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Create a custom colormap that handles NaN values
    # Use a mask for missing data
    mask = pd.isna(pvalue_matrix)
    
    # For visualization, replace NaN p-values with 1.0 (not significant)
    pvalue_display = pvalue_matrix.fillna(1.0)
    
    # Use RdYlGn_r colormap: Green for low p-values (significant), Red for high p-values
    ax = sns.heatmap(
        pvalue_display,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn_r',  # Green=low p-values (significant), Red=high p-values
        vmin=0,
        vmax=0.1,  # Focus on the 0-0.1 range where significance matters
        mask=mask,  # Mask missing data
        cbar_kws={'label': f'{metric.title()} P-Values\n(Green=Significant, Red=Not Significant)'},
        linewidths=0.5,
        square=True,
        annot_kws={'size': 9}
    )
    
    # Add effect direction markers for significant results with real data
    for i, eval_corr in enumerate(pvalue_matrix.index):
        for j, train_corr in enumerate(pvalue_matrix.columns):
            if not mask.iloc[i, j]:  # Only for real data
                effect_val = effect_matrix.iloc[i, j] if not pd.isna(effect_matrix.iloc[i, j]) else 0
                p_val = pvalue_matrix.iloc[i, j]
                
                # Add directional marker if significant (p < 0.05)
                if p_val < 0.05:
                    marker = '↑' if effect_val > 0 else '↓' if effect_val < 0 else '='
                    ax.text(j + 0.5, i + 0.15, marker, fontsize=14, fontweight='bold',
                           ha='center', va='center', color='white')
            else:
                # Mark missing data
                ax.text(j + 0.5, i + 0.5, 'N/A', fontsize=10, fontweight='bold',
                       ha='center', va='center', color='gray', alpha=0.7)
    
    # Customize the plot
    plt.title(f'{metric.title()} - Cross-Corruption P-Values\n(↑=Positive Effect, ↓=Negative Effect, N/A=Missing Data)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Training Corruption Type', fontsize=12, fontweight='bold')
    plt.ylabel('Evaluation Dataset', fontsize=12, fontweight='bold')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ {metric.title()} p-value heatmap saved to: {output_path}")

def create_effect_size_heatmap(df, metric, all_corruptions, output_path):
    """Create heatmap for effect sizes of a specific metric"""
    
    # Filter data for this metric
    metric_data = df[df['metric'] == metric].copy()
    
    # Create pivot table for effect sizes
    effect_matrix = metric_data.pivot_table(
        index='eval_corruption',
        columns='train_corruption', 
        values='effect_size',
        fill_value=0
    )
    
    # Reorder according to specified order
    ordered_corruptions = [c for c in all_corruptions if c in effect_matrix.index]
    effect_matrix = effect_matrix.reindex(index=ordered_corruptions, columns=ordered_corruptions)
    
    # Create significance matrix for annotations
    sig_matrix = metric_data.pivot_table(
        index='eval_corruption',
        columns='train_corruption',
        values='significance',
        fill_value=False
    ).reindex(index=ordered_corruptions, columns=ordered_corruptions)
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Use RdYlGn colormap: Red for negative (worse), Green for positive (better)
    max_abs_value = max(abs(effect_matrix.min().min()), abs(effect_matrix.max().max()))
    vmax = max_abs_value if max_abs_value > 0 else 1
    
    # DEBUG: Print effect matrix to verify values
    print(f"\n📊 {metric.title()} Effect Size Matrix:")
    print(effect_matrix)
    print(f"   Min value: {effect_matrix.min().min():.2f}")
    print(f"   Max value: {effect_matrix.max().max():.2f}")
    
    ax = sns.heatmap(
        effect_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',  # Red-Yellow-Green: Red=negative, Green=positive
        center=0,
        vmin=-vmax,
        vmax=vmax,
        cbar_kws={'label': f'{metric.title()} Effect Size\n(Green=Better, Red=Worse)'},
        linewidths=0.5,
        square=True,
        annot_kws={'size': 10}
    )
    
    # Add significance markers and direction arrows
    for i, eval_corr in enumerate(effect_matrix.index):
        for j, train_corr in enumerate(effect_matrix.columns):
            if eval_corr in sig_matrix.index and train_corr in sig_matrix.columns:
                if sig_matrix.loc[eval_corr, train_corr]:
                    # Add asterisk for significance
                    ax.text(j + 0.5, i + 0.15, '*', fontsize=16, fontweight='bold',
                           ha='center', va='center', color='black')
            
            # Add arrow to indicate direction (↑ positive, ↓ negative)
            if eval_corr in effect_matrix.index and train_corr in effect_matrix.columns:
                value = effect_matrix.loc[eval_corr, train_corr]
                if not np.isnan(value):
                    arrow = '↑' if value > 0 else '↓' if value < 0 else '='
                    # Use contrasting color based on background
                    arrow_color = 'darkred' if value < 0 else 'darkgreen' if value > 0 else 'gray'
                    ax.text(j + 0.5, i + 0.85, arrow, fontsize=14, fontweight='bold',
                           ha='center', va='center', color=arrow_color)
    
    # Customize the plot
    plt.title(f'{metric.title()} - Cross-Corruption Effect Sizes\n(* = Significant at α=0.05)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Training Corruption Type', fontsize=14, fontweight='bold')
    plt.ylabel('Evaluation Dataset', fontsize=14, fontweight='bold')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add grid for better readability
    ax.set_xticks(np.arange(len(effect_matrix.columns)) + 0.5, minor=True)
    ax.set_yticks(np.arange(len(effect_matrix.index)) + 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ {metric.title()} effect size heatmap saved to: {output_path}")

def create_combined_overview_heatmap(df, all_corruptions, all_metrics, output_path):
    """Create a 2x2 subplot with all metrics showing p-values"""
    
    import matplotlib.pyplot as plt_local
    from matplotlib.colors import ListedColormap
    
    fig, axes = plt_local.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    for idx, metric in enumerate(all_metrics):
        ax = axes[idx]
        
        # Get data for this metric
        metric_data = df[df['metric'] == metric].copy()
        
        # P-value matrix
        pvalue_matrix = metric_data.pivot_table(
            index='eval_corruption',
            columns='train_corruption',
            values='p_value',
            fill_value=np.nan  # Use NaN for missing data
        )
        
        # Effect size matrix for direction markers
        effect_matrix = metric_data.pivot_table(
            index='eval_corruption',
            columns='train_corruption',
            values='effect_size',
            fill_value=np.nan  # Use NaN for missing data
        )
        
        # Reorder
        ordered_corruptions = [c for c in all_corruptions if c in pvalue_matrix.index]
        pvalue_matrix = pvalue_matrix.reindex(index=ordered_corruptions, columns=ordered_corruptions)
        effect_matrix = effect_matrix.reindex(index=ordered_corruptions, columns=ordered_corruptions)
        
        # Set off-diagonal elements to NaN (only diagonal has real data)
        for i, eval_corr in enumerate(ordered_corruptions):
            for j, train_corr in enumerate(ordered_corruptions):
                if eval_corr != train_corr:
                    pvalue_matrix.loc[eval_corr, train_corr] = np.nan
                    effect_matrix.loc[eval_corr, train_corr] = np.nan
        
        # Get the RdYlGn_r colormap and set bad values (NaN) to white
        cmap = plt_local.cm.RdYlGn_r.copy()
        cmap.set_bad(color='white')
        
        # Create heatmap for p-values
        sns.heatmap(
            pvalue_matrix,
            annot=True,
            fmt='.3f',
            cmap=cmap,  # Custom colormap with white for NaN
            vmin=0,
            vmax=0.1,
            ax=ax,
            cbar_kws={'label': 'P-Value\n(White=N/A)'},
            linewidths=0.5,
            square=True,
            annot_kws={'size': 8},
            mask=pvalue_matrix.isnull()  # Mask NaN values (don't show annotation)
        )
        
        # Add direction markers for significant results (only for diagonal)
        for i, eval_corr in enumerate(pvalue_matrix.index):
            for j, train_corr in enumerate(pvalue_matrix.columns):
                if eval_corr in effect_matrix.index and train_corr in effect_matrix.columns:
                    effect_val = effect_matrix.loc[eval_corr, train_corr]
                    p_val = pvalue_matrix.loc[eval_corr, train_corr]
                    
                    # Only add markers for non-NaN values (diagonal)
                    if not np.isnan(p_val) and not np.isnan(effect_val) and p_val < 0.05:
                        marker = '↑' if effect_val > 0 else '↓' if effect_val < 0 else '='
                        ax.text(j + 0.5, i + 0.15, marker, fontsize=12, fontweight='bold',
                               ha='center', va='center', color='black')
        
        ax.set_title(f'{metric.title()}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Corruption' if idx >= 2 else '', fontsize=12)
        ax.set_ylabel('Evaluation Dataset' if idx % 2 == 0 else '', fontsize=12)
        
        # Rotate labels
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', rotation=0, labelsize=10)
    
    plt_local.suptitle('Cross-Corruption P-Values Analysis\n(Green=Significant, Red=Not Significant, White=N/A, ↑=Positive Effect, ↓=Negative Effect)', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt_local.tight_layout()
    plt_local.subplots_adjust(top=0.93)
    plt_local.savefig(output_path, dpi=300, bbox_inches='tight')
    plt_local.close()
    
    print(f"✅ Combined overview heatmap saved to: {output_path}")

def generate_cross_corruption_summary(df, output_path):
    """Generate summary statistics for cross-corruption analysis"""
    
    with open(output_path, 'w') as f:
        f.write("=== CROSS-CORRUPTION ANALYSIS SUMMARY ===\n\n")
        
        f.write(f"Total Cross-Corruption Comparisons: {len(df)}\n")
        f.write(f"Metrics Analyzed: {', '.join(sorted(df['metric'].unique()))}\n")
        f.write(f"Training Corruptions: {', '.join(sorted(df['train_corruption'].unique()))}\n")
        f.write(f"Evaluation Datasets: {', '.join(sorted(df['eval_corruption'].unique()))}\n\n")
        
        f.write("=== EFFECT DIRECTION SUMMARY ===\n")
        positive_effects = (df['effect_size'] > 0).sum()
        negative_effects = (df['effect_size'] < 0).sum()
        neutral_effects = (df['effect_size'] == 0).sum()
        
        f.write(f"Positive Effects (Better Performance): {positive_effects}\n")
        f.write(f"Negative Effects (Worse Performance): {negative_effects}\n")
        f.write(f"Neutral Effects: {neutral_effects}\n\n")
        
        f.write("=== STRONGEST POSITIVE EFFECTS ===\n")
        positive_df = df[df['effect_size'] > 0].sort_values('effect_size', ascending=False).head(10)
        for _, row in positive_df.iterrows():
            f.write(f"{row['train_corruption']} → {row['eval_corruption']} ({row['metric']}): "
                   f"+{row['effect_size']:.2f} (p={row['p_value']:.4f})\n")
        
        f.write("\n=== STRONGEST NEGATIVE EFFECTS ===\n")
        negative_df = df[df['effect_size'] < 0].sort_values('effect_size', ascending=True).head(10)
        for _, row in negative_df.iterrows():
            f.write(f"{row['train_corruption']} → {row['eval_corruption']} ({row['metric']}): "
                   f"{row['effect_size']:.2f} (p={row['p_value']:.4f})\n")
    
    print(f"✅ Cross-corruption summary saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate cross-corruption heatmaps")
    parser.add_argument('--input',
                       default='results/statistical_log.txt',
                       help='Path to statistical_log.txt file')
    parser.add_argument('--output_dir',
                       default='results/cross_corruption_analysis',
                       help='Output directory for heatmaps')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse the statistical log file
    print(f"📖 Parsing statistical log: {args.input}")
    df = parse_statistical_log_cross_corruption(args.input)
    
    # Create full cross-corruption matrix
    print("🔄 Creating cross-corruption matrix...")
    full_df, all_corruptions, all_metrics = create_cross_corruption_matrix_data(df)
    
    print(f"📊 Cross-corruption analysis setup:")
    print(f"   - Training corruptions: {all_corruptions}")
    print(f"   - Evaluation datasets: {all_corruptions}")
    print(f"   - Metrics: {all_metrics}")
    print(f"   - Total combinations: {len(full_df)}")
    
    # Generate individual metric heatmaps
    print("\n🎨 Generating P-VALUE heatmaps...")
    for metric in all_metrics:
        output_file = output_dir / f'{metric}_pvalue_heatmap.png'
        create_metric_heatmap(full_df, metric, all_corruptions, output_file)
    
    # Generate effect size heatmaps
    print("\n🎨 Generating EFFECT SIZE heatmaps...")
    for metric in all_metrics:
        output_file = output_dir / f'{metric}_effect_size_heatmap.png'
        create_effect_size_heatmap(full_df, metric, all_corruptions, output_file)
    
    # Generate combined overview
    print("\n🎨 Generating combined P-VALUE overview...")
    create_combined_overview_heatmap(full_df, all_corruptions, all_metrics, 
                                   output_dir / 'combined_pvalue_heatmap.png')
    
    # Generate summary
    generate_cross_corruption_summary(full_df, output_dir / 'cross_corruption_summary.txt')
    
    print(f"\n🎉 Cross-corruption analysis complete!")
    print(f"📁 Results saved to: {output_dir}")
    print("\nGenerated files:")
    print("🔴 P-VALUE HEATMAPS (Green=Significant, Red=Not Significant):")
    for metric in all_metrics:
        print(f"  📈 {metric}_pvalue_heatmap.png")
    print("🟢 EFFECT SIZE HEATMAPS (Green=Better Performance, Red=Worse Performance):")
    for metric in all_metrics:
        print(f"  📊 {metric}_effect_size_heatmap.png")
    print("📋 combined_pvalue_heatmap.png")
    print("📄 cross_corruption_summary.txt")

if __name__ == '__main__':
    main()
