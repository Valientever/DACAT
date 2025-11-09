#!/usr/bin/env python3
"""
Create clear bar graphs showing:
1. Absolute performance values
2. Performance differences between clean and corruption-trained models
3. Statistical significance markers
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Sample data from your statistical_log_ext.txt
# Replace this with actual data parsing
data = {
    'gaussian_noise': {
        'clean': {'accuracy': 37.68, 'jaccard': 11.37, 'precision': 39.00, 'recall': 23.44},
        'trained': {'accuracy': 14.66, 'jaccard': 6.56, 'precision': 36.98, 'recall': 17.82},
        'p_values': {'accuracy': 0.0000, 'jaccard': 0.0001, 'precision': 0.4291, 'recall': 0.0003}
    },
    'motion_blur': {
        'clean': {'accuracy': 8.21, 'jaccard': 6.05, 'precision': 36.54, 'recall': 15.49},
        'trained': {'accuracy': 12.85, 'jaccard': 8.68, 'precision': 53.28, 'recall': 15.31},
        'p_values': {'accuracy': 0.0012, 'jaccard': 0.0001, 'precision': 0.0024, 'recall': 0.7257}
    },
    'defocus_blur': {
        'clean': {'accuracy': 45.50, 'jaccard': 10.89, 'precision': 50.87, 'recall': 19.16},
        'trained': {'accuracy': 8.69, 'jaccard': 5.34, 'precision': 35.70, 'recall': 16.65},
        'p_values': {'accuracy': 0.0000, 'jaccard': 0.0000, 'precision': 0.0192, 'recall': 0.0787}
    },
    'smoke_effect': {
        'clean': {'accuracy': 25.33, 'jaccard': 9.51, 'precision': 43.35, 'recall': 16.18},
        'trained': {'accuracy': 34.18, 'jaccard': 8.32, 'precision': 46.64, 'recall': 12.56},
        'p_values': {'accuracy': 0.0072, 'jaccard': 0.3305, 'precision': 0.7048, 'recall': 0.6634}
    },
    'uneven_illumination': {
        'clean': {'accuracy': 24.08, 'jaccard': 8.48, 'precision': 31.47, 'recall': 25.31},
        'trained': {'accuracy': 5.47, 'jaccard': 5.42, 'precision': 31.22, 'recall': 20.64},
        'p_values': {'accuracy': 0.0000, 'jaccard': 0.0035, 'precision': 0.4223, 'recall': 0.7898}
    },
    'random': {
        'clean': {'accuracy': 10.91, 'jaccard': 6.22, 'precision': 24.21, 'recall': 20.29},
        'trained': {'accuracy': 6.79, 'jaccard': 4.71, 'precision': 21.78, 'recall': 18.25},
        'p_values': {'accuracy': 0.0005, 'jaccard': 0.9664, 'precision': 0.6033, 'recall': 0.4389}
    }
}

def get_significance_marker(p_value):
    """Convert p-value to significance marker"""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'

def create_comparison_bar_graph(data, metric='accuracy', output_path='comparison_bars.png'):
    """
    Create side-by-side bar graphs with values and differences
    """
    corruptions = list(data.keys())
    corruption_labels = [c.replace('_', ' ').title() for c in corruptions]
    
    clean_values = [data[c]['clean'][metric] for c in corruptions]
    trained_values = [data[c]['trained'][metric] for c in corruptions]
    differences = [trained_values[i] - clean_values[i] for i in range(len(corruptions))]
    p_values = [data[c]['p_values'][metric] for c in corruptions]
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- LEFT PLOT: Absolute Values ---
    x = np.arange(len(corruptions))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, clean_values, width, label='Clean-Trained Model', 
                    color='#3498db', edgecolor='black', linewidth=1.2)
    bars2 = ax1.bar(x + width/2, trained_values, width, label='Corruption-Trained Model',
                    color='#e74c3c', edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add significance markers
    max_height = max(max(clean_values), max(trained_values))
    for i, (p_val, diff) in enumerate(zip(p_values, differences)):
        sig = get_significance_marker(p_val)
        if sig != 'ns':
            # Position significance marker above the taller bar
            y_pos = max(clean_values[i], trained_values[i]) + max_height * 0.05
            ax1.text(i, y_pos, sig, ha='center', va='bottom', 
                    fontsize=14, fontweight='bold', color='green' if diff > 0 else 'red')
    
    ax1.set_xlabel('Corruption Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel(f'{metric.title()} (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{metric.title()} Comparison: Clean vs Corruption-Trained Models',
                 fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(corruption_labels, rotation=45, ha='right')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max_height * 1.2)
    
    # --- RIGHT PLOT: Differences ---
    colors = ['green' if d > 0 else 'red' for d in differences]
    bars3 = ax2.bar(x, differences, color=colors, alpha=0.7, edgecolor='black', linewidth=1.2)
    
    # Add value labels and significance
    for i, (bar, diff, p_val) in enumerate(zip(bars3, differences, p_values)):
        height = bar.get_height()
        sig = get_significance_marker(p_val)
        
        # Position text
        va = 'bottom' if height > 0 else 'top'
        y_offset = 0.5 if height > 0 else -0.5
        
        # Show difference value
        ax2.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                f'{diff:+.1f}%',
                ha='center', va=va, fontsize=10, fontweight='bold')
        
        # Show significance
        if sig != 'ns':
            y_sig = height + (1.5 if height > 0 else -1.5)
            ax2.text(bar.get_x() + bar.get_width()/2., y_sig,
                    sig, ha='center', va=va, fontsize=12, fontweight='bold',
                    color='darkgreen' if height > 0 else 'darkred')
    
    # Add zero line
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=2)
    
    ax2.set_xlabel('Corruption Type', fontsize=12, fontweight='bold')
    ax2.set_ylabel(f'{metric.title()} Difference (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Performance Change with Corruption Training\n(Positive = Improvement)',
                 fontsize=14, fontweight='bold', pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(corruption_labels, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add legend for significance
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Improvement'),
        Patch(facecolor='red', alpha=0.7, label='Degradation'),
    ]
    ax2.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    # Add significance note at bottom
    fig.text(0.5, 0.02, '* p<0.05, ** p<0.01, *** p<0.001, ns = not significant',
            ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def create_all_metrics_panel(data, output_path='all_metrics_comparison.png'):
    """
    Create a 2x2 panel showing all 4 metrics
    """
    metrics = ['accuracy', 'jaccard', 'precision', 'recall']
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        corruptions = list(data.keys())
        corruption_labels = [c.replace('_', ' ').title() for c in corruptions]
        
        clean_values = [data[c]['clean'][metric] for c in corruptions]
        trained_values = [data[c]['trained'][metric] for c in corruptions]
        differences = [trained_values[i] - clean_values[i] for i in range(len(corruptions))]
        p_values = [data[c]['p_values'][metric] for c in corruptions]
        
        x = np.arange(len(corruptions))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, clean_values, width, label='Clean-Trained', 
                      color='#3498db', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, trained_values, width, label='Corruption-Trained',
                      color='#e74c3c', alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=8)
        
        # Add significance markers
        max_height = max(max(clean_values), max(trained_values))
        for i, (p_val, diff) in enumerate(zip(p_values, differences)):
            sig = get_significance_marker(p_val)
            if sig != 'ns':
                y_pos = max(clean_values[i], trained_values[i]) + max_height * 0.05
                ax.text(i, y_pos, sig, ha='center', va='bottom', 
                       fontsize=11, fontweight='bold',
                       color='green' if diff > 0 else 'red')
        
        ax.set_xlabel('Corruption Type', fontsize=10, fontweight='bold')
        ax.set_ylabel(f'{metric.title()} (%)', fontsize=10, fontweight='bold')
        ax.set_title(f'{metric.title()}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(corruption_labels, rotation=45, ha='right', fontsize=9)
        
        if idx == 0:
            ax.legend(loc='upper left', fontsize=9)
        
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, max_height * 1.15)
    
    fig.suptitle('Performance Comparison Across All Metrics\nClean-Trained vs Corruption-Trained Models',
                fontsize=16, fontweight='bold', y=0.995)
    fig.text(0.5, 0.01, '* p<0.05, ** p<0.01, *** p<0.001, ns = not significant',
            ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.99])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def create_heatmap_with_values(data, output_path='heatmap_differences.png'):
    """
    Create a heatmap showing performance differences
    """
    corruptions = list(data.keys())
    metrics = ['accuracy', 'jaccard', 'precision', 'recall']
    
    # Create matrix of differences
    diff_matrix = []
    annot_matrix = []
    
    for corruption in corruptions:
        row_diffs = []
        row_annots = []
        for metric in metrics:
            clean = data[corruption]['clean'][metric]
            trained = data[corruption]['trained'][metric]
            diff = trained - clean
            p_val = data[corruption]['p_values'][metric]
            sig = get_significance_marker(p_val)
            
            row_diffs.append(diff)
            row_annots.append(f'{diff:+.1f}\n{sig}')
        
        diff_matrix.append(row_diffs)
        annot_matrix.append(row_annots)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(diff_matrix, cmap='RdYlGn', aspect='auto', 
                   vmin=-40, vmax=40, alpha=0.8)
    
    # Set ticks
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(corruptions)))
    ax.set_xticklabels([m.title() for m in metrics], fontsize=11, fontweight='bold')
    ax.set_yticklabels([c.replace('_', ' ').title() for c in corruptions], 
                       fontsize=11, fontweight='bold')
    
    # Add annotations
    for i in range(len(corruptions)):
        for j in range(len(metrics)):
            text = ax.text(j, i, annot_matrix[i][j],
                          ha="center", va="center", color="black",
                          fontsize=10, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Performance Change (%)', rotation=270, labelpad=20, 
                   fontsize=11, fontweight='bold')
    
    ax.set_title('Performance Change with Corruption Training\n(Green = Improvement, Red = Degradation)',
                fontsize=13, fontweight='bold', pad=20)
    
    fig.text(0.5, 0.02, '* p<0.05, ** p<0.01, *** p<0.001, ns = not significant',
            ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

# Main execution
if __name__ == '__main__':
    output_dir = Path('/home/santhi/Documents/DACAT/src/Cholec80/results/bar_graphs')
    output_dir.mkdir(exist_ok=True)
    
    print("🎨 Generating bar graphs...")
    print("=" * 60)
    
    # Create individual metric comparisons
    for metric in ['accuracy', 'jaccard', 'precision', 'recall']:
        output_path = output_dir / f'{metric}_comparison.png'
        create_comparison_bar_graph(data, metric=metric, output_path=str(output_path))
    
    # Create all-metrics panel
    create_all_metrics_panel(data, output_path=str(output_dir / 'all_metrics_panel.png'))
    
    # Create heatmap with values
    create_heatmap_with_values(data, output_path=str(output_dir / 'heatmap_differences.png'))
    
    print("=" * 60)
    print(f"✅ All graphs saved to: {output_dir}")
    print("\nGenerated files:")
    print("  1. accuracy_comparison.png - Accuracy bars + differences")
    print("  2. jaccard_comparison.png - Jaccard bars + differences")
    print("  3. precision_comparison.png - Precision bars + differences")
    print("  4. recall_comparison.png - Recall bars + differences")
    print("  5. all_metrics_panel.png - 2x2 panel with all metrics")
    print("  6. heatmap_differences.png - Heatmap showing changes")
