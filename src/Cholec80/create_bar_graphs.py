#!/usr/bin/env python3
"""
Create clear bar graphs showing:
1. Performance differences between clean and corruption-trained models
2. Error bars showing 95% confidence intervals
3. Statistical significance markers

Parses data from statistical_log_with_ci.txt
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import re

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def parse_statistical_log(file_path):
    """
    Parse statistical_log_with_ci.txt and extract comparison data.
    
    Returns:
        DataFrame with columns: corruption, metric, median, difference, ci_lower, ci_upper, 
                                p_value, significance
    """
    data_rows = []
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract alpha level
    alpha_match = re.search(r'Alpha\s*=\s*([\d.]+)', content)
    alpha = float(alpha_match.group(1)) if alpha_match else 0.05
    
    # Split by corruption sections
    corruption_sections = re.split(r'===\s*Corruption:\s*(.+?)\s*===', content)[1:]
    
    print("📖 Parsing statistical log with confidence intervals...")
    
    # Process pairs: (corruption_name, section_content)
    for i in range(0, len(corruption_sections), 2):
        corruption = corruption_sections[i].strip()
        section_content = corruption_sections[i + 1]
        
        # Split into lines and find data lines
        lines = section_content.strip().split('\n')
        data_lines = [line for line in lines if line.strip() and 
                     not line.startswith('#') and 
                     not line.startswith('metric') and
                     (line.split()[0] != 'metric' if line.split() else True)]
        
        for line in data_lines:
            parts = line.split()
            if len(parts) >= 11:  # Updated to match new format with CI columns
                metric = parts[0]
                pairs = int(parts[1])
                median = float(parts[2])
                train_condition = parts[3]
                eval_condition = parts[4]
                difference = float(parts[5])
                ci_lower = float(parts[6])
                ci_upper = float(parts[7])
                wilcoxon_stat = float(parts[8])
                p_value = float(parts[9])
                significance = parts[10] == 'True'
                
                data_rows.append({
                    'corruption': corruption,
                    'metric': metric,
                    'median': median,
                    'difference': difference,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'p_value': p_value,
                    'significance': significance,
                    'alpha': alpha
                })
    
    df = pd.DataFrame(data_rows)
    print(f"✅ Extracted {len(df)} comparisons from statistical log")
    print(f"   Corruptions: {df['corruption'].unique().tolist()}")
    print(f"   Metrics: {df['metric'].unique().tolist()}")
    
    return df

def get_significance_marker(p_value):
    """Convert p-value to significance marker"""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''


def create_grouped_bar_chart_with_error_bars(df, output_path='grouped_bar_chart.png'):
    """
    Create a grouped bar chart showing median differences across all metrics and corruptions,
    with error bars representing 95% confidence intervals and significance indicators.
    
    X-axis: Metric Type (Accuracy, Jaccard, Precision, Recall)
    Y-axis: Median % Difference (Trained Model vs. Baseline)
    Groups: Different corruptions (color-coded)
    Error bars: 95% CI
    Asterisks: Statistical significance (p < 0.05)
    """
    # Define corruption order and colors
    corruption_order = [ 'gaussian_noise', 'motion_blur', 'defocus_blur',
                       'smoke_effect', 'uneven_illumination', 'random' ]
    
    # Filter to only include corruptions that exist in data
    available_corruptions = [c for c in corruption_order if c in df['corruption'].unique()]
    
    # Define bright colors for each corruption
    colors = {
        'defocus_blur': '#FF6B6B',        # Bright Red
        'gaussian_noise': '#4ECDC4',      # Bright Cyan
        'motion_blur': '#95E1D3',         # Bright Mint Green
        'random': '#FFD93D',              # Bright Yellow
        'smoke_effect': '#C44569',        # Bright Magenta/Pink
        'uneven_illumination': '#6BCB77'  # Bright Green
    }
    
    # Metrics order
    metrics = ['accuracy', 'precision', 'recall', 'jaccard']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'Jaccard']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set up bar positions
    x = np.arange(len(metrics))
    width = 0.13  # Width of each bar
    n_corruptions = len(available_corruptions)
    
    # Calculate offset for each corruption
    offsets = np.linspace(-(n_corruptions-1)*width/2, (n_corruptions-1)*width/2, n_corruptions)
    
    # Plot bars for each corruption
    for idx, corruption in enumerate(available_corruptions):
        corruption_data = df[df['corruption'] == corruption]
        
        differences = []
        errors_lower = []
        errors_upper = []
        p_values = []
        
        for metric in metrics:
            metric_data = corruption_data[corruption_data['metric'] == metric]
            if not metric_data.empty:
                row = metric_data.iloc[0]
                differences.append(row['difference'])
                # Calculate error bar distances from median difference
                # Handle edge case where bootstrap CI bounds may not perfectly contain the median
                err_lower = row['difference'] - row['ci_lower']
                err_upper = row['ci_upper'] - row['difference']
                
                # Ensure error bars are non-negative (data reporting issue, not visualization issue)
                # This can happen with bootstrap when median falls outside percentile CI
                err_lower = max(0, err_lower)
                err_upper = max(0, err_upper)
                
                if err_lower == 0 or err_upper == 0:
                    print(f"⚠️  Note: Adjusted error bar for {corruption}/{metric} " +
                          f"(diff={row['difference']:.2f}, CI=[{row['ci_lower']:.2f}, {row['ci_upper']:.2f}])")
                
                errors_lower.append(err_lower)
                errors_upper.append(err_upper)
                p_values.append(row['p_value'])
            else:
                differences.append(0)
                errors_lower.append(0)
                errors_upper.append(0)
                p_values.append(1.0)
        
        # Combine lower and upper errors for error bars
        errors = [errors_lower, errors_upper]
        
        # Create bars
        corruption_label = corruption.replace('_', ' ').title()
        bars = ax.bar(x + offsets[idx], differences, width, 
                     label=corruption_label,
                     color=colors.get(corruption, '#95a5a6'),
                     alpha=0.85,
                     edgecolor='black',
                     linewidth=0.8,
                     yerr=errors,
                     capsize=3,
                     error_kw={'linewidth': 1.5, 'ecolor': 'black', 'alpha': 0.7})
        
        # Add difference values and significance markers
        for i, (bar, p_val, diff) in enumerate(zip(bars, p_values, differences)):
            error_top = diff + errors_upper[i]
            error_bottom = diff - errors_lower[i]
            sig_marker = get_significance_marker(p_val)
            
            if diff > 0:
                # For positive bars, place value on top of error bar
                value_y = error_top + 1.5
                value_va = 'bottom'
            else:
                # For negative bars, place value below error bar
                value_y = error_bottom - 1.5
                value_va = 'top'
            
            # Add the difference value
            ax.text(bar.get_x() + bar.get_width()/2, value_y,
                   f'{diff:.1f}',
                   ha='center', va=value_va,
                   fontsize=9, fontweight='bold',
                   color='black')
            
            # Add significance marker above/below the value
            if sig_marker:
                if diff > 0:
                    sig_y = error_top + 5
                    sig_va = 'bottom'
                else:
                    sig_y = error_bottom - 5
                    sig_va = 'top'
                
                ax.text(bar.get_x() + bar.get_width()/2, sig_y,
                       sig_marker,
                       ha='center', va=sig_va,
                       fontsize=10, fontweight='bold',
                       color='red')
    
    # Customize plot
    ax.set_xlabel('Metric Type', fontsize=13, fontweight='bold')
    ax.set_ylabel('Median % Difference (Trained Model vs. Baseline)', fontsize=13, fontweight='bold')
    ax.set_title('Comparison of Model Performance  Across Various Corruptions\n' +
                #  '(Corruption Training vs. Clean Training on Corrupted Data)',
                # '(Internal_validation)',
                '(External_validation)',
                 fontsize=15, fontweight='bold', pad=50)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.3)
    
    # Move legend to top in a single row (adjusted position to avoid overlap)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), 
             fontsize=12, ncol=6, framealpha=0.95, borderaxespad=0,
             columnspacing=1.0, handlelength=1.5)
    
    # Set y-axis range from -50 to 100 with 10-unit increments for better visibility
    ax.set_ylim(-50, 100)
    ax.set_yticks(np.arange(-50, 101, 10))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add footnote for significance
    fig.text(0.5, 0.02, 
             '* p<0.05, ** p<0.01, *** p<0.001 (Wilcoxon signed-rank test)\n' +
             'Error bars represent 95% confidence intervals (10,000 bootstrap iterations)',
             ha='center', fontsize=9, style='italic')
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved grouped bar chart: {output_path}")
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
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate bar graphs with error bars from statistical_log_with_ci.txt'
    )
    parser.add_argument('--input', '-i', 
                       default='results/statistical_log_with_ci.txt',
                       help='Path to statistical_log_with_ci.txt file (default: results/statistical_log_with_ci.txt)')
    parser.add_argument('--output_dir', '-o',
                       default='results/ext_bar_graphs_CI_spacing',
                       help='Output directory for graphs (default: results/ext_bar_graphs_CI_spacing)')
    
    args = parser.parse_args()
    
    # Setup paths
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎨 Generating bar graphs with error bars and significance indicators...")
    print("=" * 70)
    print(f"📂 Input file: {input_path}")
    print(f"📂 Output directory: {output_dir}")
    print("=" * 70)
    
    # Parse the statistical log file
    df = parse_statistical_log(input_path)
    
    # Create grouped bar chart with error bars (main visualization)
    print("\n📊 Creating grouped bar chart with error bars...")
    create_grouped_bar_chart_with_error_bars(df, output_path=str(output_dir / 'grouped_bar_chart_with_ci.png'))
    
    print("\n" + "=" * 70)
    print(f"✅ All graphs saved to: {output_dir}")
    print("\nGenerated files:")
    print("  📊 grouped_bar_chart_with_ci.png - Main grouped visualization with error bars")
