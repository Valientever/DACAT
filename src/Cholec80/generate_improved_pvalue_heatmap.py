#!/usr/bin/env python3
"""
generate_improved_pvalue_heatmap.py

Creates improved p-value heatmaps that clearly show:
1. Effect direction (positive/negative) using color
2. Statistical significance using border/marker
3. White cells for missing data

Key improvements:
- RED cells = Negative effect (degradation)
- GREEN cells = Positive effect (improvement)
- Color intensity = magnitude of effect
- Cell border thickness = significance level
- Asterisks (*) for p<0.05, (**) for p<0.01, (***) for p<0.001
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def parse_statistical_log(file_path):
    """Parse statistical_log.txt and extract comparison data"""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    alpha_match = re.search(r'Alpha[=:]\s*([\d.]+)', content)
    alpha = float(alpha_match.group(1)) if alpha_match else 0.05
    
    corruption_sections = re.split(r'=== Corruption: (.+?) ===', content)[1:]
    
    data_rows = []
    
    print("📖 Parsing statistical log...")
    
    for i in range(0, len(corruption_sections), 2):
        eval_corruption = corruption_sections[i].strip()
        section_content = corruption_sections[i + 1]
        
        lines = section_content.strip().split('\n')
        data_lines = [line for line in lines if line.strip() and not line.startswith('#') and 'metric' not in line]
        
        for line in data_lines:
            parts = line.split()
            if len(parts) >= 8:
                metric = parts[0]
                pairs = int(parts[1])
                median = float(parts[2])
                train_corruption = eval_corruption
                difference = float(parts[-4])
                wilcoxon_stat = float(parts[-3])
                p_value = float(parts[-2])
                significance = parts[-1] == 'True'
                
                data_rows.append({
                    'train_corruption': train_corruption,
                    'eval_corruption': eval_corruption,
                    'metric': metric,
                    'pairs': pairs,
                    'median': median,
                    'effect_size': difference,
                    'wilcoxon_stat': wilcoxon_stat,
                    'p_value': p_value,
                    'significance': significance,
                    'alpha': alpha
                })
    
    df = pd.DataFrame(data_rows)
    print(f"✅ Extracted {len(df)} comparisons")
    return df

def get_significance_stars(p_value):
    """Convert p-value to significance stars"""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''

def create_effect_direction_heatmap(df, all_corruptions, all_metrics, output_path):
    """
    Create heatmap where:
    - Cell COLOR indicates effect DIRECTION (red=negative, green=positive)
    - Cell border/annotation indicates SIGNIFICANCE
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(22, 18))
    axes = axes.flatten()
    
    for idx, metric in enumerate(all_metrics):
        ax = axes[idx]
        
        metric_data = df[df['metric'] == metric].copy()
        
        # Create matrices
        effect_matrix = metric_data.pivot_table(
            index='eval_corruption',
            columns='train_corruption',
            values='effect_size',
            fill_value=np.nan
        )
        
        pvalue_matrix = metric_data.pivot_table(
            index='eval_corruption',
            columns='train_corruption',
            values='p_value',
            fill_value=np.nan
        )
        
        # Reorder
        ordered_corruptions = [c for c in all_corruptions if c in effect_matrix.index]
        effect_matrix = effect_matrix.reindex(index=ordered_corruptions, columns=ordered_corruptions)
        pvalue_matrix = pvalue_matrix.reindex(index=ordered_corruptions, columns=ordered_corruptions)
        
        # Set off-diagonal to NaN (only diagonal has data)
        for i, eval_corr in enumerate(ordered_corruptions):
            for j, train_corr in enumerate(ordered_corruptions):
                if eval_corr != train_corr:
                    effect_matrix.loc[eval_corr, train_corr] = np.nan
                    pvalue_matrix.loc[eval_corr, train_corr] = np.nan
        
        # Create diverging colormap: Red (negative) -> White (zero) -> Green (positive)
        from matplotlib.colors import TwoSlopeNorm
        
        # Get max absolute value for symmetric scale
        valid_effects = effect_matrix.values[~np.isnan(effect_matrix.values)]
        if len(valid_effects) > 0:
            vmax = max(abs(valid_effects.min()), abs(valid_effects.max()))
        else:
            vmax = 1
        
        # Create custom colormap
        cmap = plt.cm.RdYlGn.copy()  # Red-Yellow-Green
        cmap.set_bad(color='white')  # NaN = white
        
        # Create heatmap with diverging colors centered at 0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        
        im = ax.imshow(effect_matrix.values, cmap=cmap, norm=norm, aspect='auto')
        
        # Add cell borders based on significance
        for i in range(len(ordered_corruptions)):
            for j in range(len(ordered_corruptions)):
                eval_corr = ordered_corruptions[i]
                train_corr = ordered_corruptions[j]
                
                if eval_corr in effect_matrix.index and train_corr in effect_matrix.columns:
                    effect_val = effect_matrix.loc[eval_corr, train_corr]
                    p_val = pvalue_matrix.loc[eval_corr, train_corr]
                    
                    if not np.isnan(effect_val) and not np.isnan(p_val):
                        # Show effect value
                        text_color = 'white' if abs(effect_val) > vmax * 0.5 else 'black'
                        ax.text(j, i + 0.1, f'{effect_val:.1f}',
                               ha='center', va='center', color=text_color,
                               fontsize=11, fontweight='bold')
                        
                        # Show significance stars
                        stars = get_significance_stars(p_val)
                        if stars:
                            ax.text(j, i - 0.25, stars,
                                   ha='center', va='center', color='black',
                                   fontsize=14, fontweight='bold')
                        
                        # Add thick border for significant results
                        if p_val < 0.05:
                            rect = plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                               fill=False, edgecolor='black',
                                               linewidth=3)
                            ax.add_patch(rect)
                    else:
                        # N/A text for missing data
                        ax.text(j, i, 'N/A',
                               ha='center', va='center', color='gray',
                               fontsize=10, style='italic')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(ordered_corruptions)))
        ax.set_yticks(np.arange(len(ordered_corruptions)))
        ax.set_xticklabels([c.replace('_', ' ').title() for c in ordered_corruptions],
                          rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels([c.replace('_', ' ').title() for c in ordered_corruptions],
                          fontsize=10)
        
        # Grid lines
        ax.set_xticks(np.arange(len(ordered_corruptions)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(ordered_corruptions)) - 0.5, minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
        
        ax.set_title(f'{metric.title()}', fontsize=15, fontweight='bold', pad=15)
        
        if idx >= 2:
            ax.set_xlabel('Training Corruption Type', fontsize=12, fontweight='bold')
        if idx % 2 == 0:
            ax.set_ylabel('Evaluation Dataset', fontsize=12, fontweight='bold')
        
        # Add colorbar for this subplot
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Effect Size\n(Red=Worse, Green=Better)', rotation=270, 
                      labelpad=20, fontsize=10)
    
    # Main title
    fig.suptitle('Performance Change with Corruption Training\n' +
                 '(Cell Color: Red=Degradation, Green=Improvement | ' +
                 'Black Border: Significant p<0.05 | *p<0.05, **p<0.01, ***p<0.001)',
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Effect direction heatmap saved: {output_path}")

def create_simplified_heatmap(df, all_corruptions, metric, output_path):
    """
    Create single-metric heatmap with clear visual coding:
    - Background color = effect direction and magnitude
    - Text = effect size value
    - Stars = significance level
    """
    
    metric_data = df[df['metric'] == metric].copy()
    
    effect_matrix = metric_data.pivot_table(
        index='eval_corruption',
        columns='train_corruption',
        values='effect_size',
        fill_value=np.nan
    )
    
    pvalue_matrix = metric_data.pivot_table(
        index='eval_corruption',
        columns='train_corruption',
        values='p_value',
        fill_value=np.nan
    )
    
    ordered_corruptions = [c for c in all_corruptions if c in effect_matrix.index]
    effect_matrix = effect_matrix.reindex(index=ordered_corruptions, columns=ordered_corruptions)
    pvalue_matrix = pvalue_matrix.reindex(index=ordered_corruptions, columns=ordered_corruptions)
    
    # Set off-diagonal to NaN
    for i, eval_corr in enumerate(ordered_corruptions):
        for j, train_corr in enumerate(ordered_corruptions):
            if eval_corr != train_corr:
                effect_matrix.loc[eval_corr, train_corr] = np.nan
                pvalue_matrix.loc[eval_corr, train_corr] = np.nan
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get max for symmetric scale
    valid_effects = effect_matrix.values[~np.isnan(effect_matrix.values)]
    vmax = max(abs(valid_effects.min()), abs(valid_effects.max())) if len(valid_effects) > 0 else 1
    
    # Diverging colormap
    from matplotlib.colors import TwoSlopeNorm
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color='white')
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    im = ax.imshow(effect_matrix.values, cmap=cmap, norm=norm, aspect='auto')
    
    # Annotations
    for i in range(len(ordered_corruptions)):
        for j in range(len(ordered_corruptions)):
            eval_corr = ordered_corruptions[i]
            train_corr = ordered_corruptions[j]
            
            if eval_corr in effect_matrix.index and train_corr in effect_matrix.columns:
                effect_val = effect_matrix.loc[eval_corr, train_corr]
                p_val = pvalue_matrix.loc[eval_corr, train_corr]
                
                if not np.isnan(effect_val):
                    # Effect value
                    text_color = 'white' if abs(effect_val) > vmax * 0.6 else 'black'
                    ax.text(j, i, f'{effect_val:.1f}',
                           ha='center', va='center', color=text_color,
                           fontsize=13, fontweight='bold')
                    
                    # Significance stars below
                    stars = get_significance_stars(p_val)
                    if stars:
                        ax.text(j, i + 0.35, stars,
                               ha='center', va='center', color='black',
                               fontsize=16, fontweight='bold')
                else:
                    ax.text(j, i, 'N/A',
                           ha='center', va='center', color='gray',
                           fontsize=11, style='italic')
    
    # Labels
    ax.set_xticks(np.arange(len(ordered_corruptions)))
    ax.set_yticks(np.arange(len(ordered_corruptions)))
    ax.set_xticklabels([c.replace('_', ' ').title() for c in ordered_corruptions],
                      rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels([c.replace('_', ' ').title() for c in ordered_corruptions],
                      fontsize=11)
    
    # Grid
    ax.set_xticks(np.arange(len(ordered_corruptions)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(ordered_corruptions)) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=1.5)
    
    ax.set_xlabel('Training Corruption Type', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('Evaluation Dataset', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_title(f'{metric.title()} - Performance Change with Corruption Training\n' +
                 '(Green=Improvement, Red=Degradation, *p<0.05, **p<0.01, ***p<0.001)',
                 fontsize=14, fontweight='bold', pad=20)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Effect Size (%)', rotation=270, labelpad=20, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ {metric.title()} simplified heatmap saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate improved p-value heatmaps')
    parser.add_argument('--input', default='results/statistical_log.txt',
                       help='Path to statistical_log.txt')
    parser.add_argument('--output_dir', default='results/improved_heatmaps',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Parse data
    df = parse_statistical_log(args.input)
    
    # Setup output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define order
    user_specified_order = [
        'gaussian_noise',
        'motion_blur',
        'defocus_blur',
        'smoke_effect',
        'uneven_illumination',
        'random'
    ]
    
    available_corruptions = list(df['train_corruption'].unique())
    ordered_corruptions = [c for c in user_specified_order if c in available_corruptions]
    all_metrics = sorted(df['metric'].unique())
    
    print(f"\n🎨 Generating improved visualizations...")
    print(f"   Corruptions: {ordered_corruptions}")
    print(f"   Metrics: {all_metrics}")
    
    # Create combined 2x2 heatmap
    create_effect_direction_heatmap(df, ordered_corruptions, all_metrics,
                                   output_dir / 'combined_effect_direction_heatmap.png')
    
    # Create individual metric heatmaps
    for metric in all_metrics:
        create_simplified_heatmap(df, ordered_corruptions, metric,
                                 output_dir / f'{metric}_effect_heatmap.png')
    
    print(f"\n✅ All improved heatmaps saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  📊 combined_effect_direction_heatmap.png - All metrics in one view")
    for metric in all_metrics:
        print(f"  📈 {metric}_effect_heatmap.png - Detailed {metric} visualization")

if __name__ == '__main__':
    main()
