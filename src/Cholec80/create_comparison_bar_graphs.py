#!/usr/bin/env python3
"""
Create comparison bar graphs showing:
1. Clean-trained model median performance (baseline)
2. Corruption-trained model median performance (trained)
3. Side-by-side comparison for each corruption and metric

Parses data from statistical_log_with_clean_median.txt
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
    Parse statistical_log_with_clean_median.txt and extract comparison data.
    
    Returns:
        DataFrame with columns: corruption, metric, median_baseline, median_trained,
                                difference, ci_lower, ci_upper, p_value, significance
    """
    data_rows = []
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract alpha level
    alpha_match = re.search(r'Alpha\s*=\s*([\d.]+)', content)
    alpha = float(alpha_match.group(1)) if alpha_match else 0.05
    
    # Split by corruption sections
    corruption_sections = re.split(r'===\s*Corruption:\s*(.+?)\s*===', content)[1:]
    
    print("📖 Parsing statistical log with baseline and trained medians...")
    
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
            if len(parts) >= 12:  # Updated to match new format with baseline and trained columns
                metric = parts[0]
                pairs = int(parts[1])
                median_baseline = float(parts[2])
                median_trained = float(parts[3])
                train_condition = parts[4]
                eval_condition = parts[5]
                difference = float(parts[6])
                ci_lower = float(parts[7])
                ci_upper = float(parts[8])
                wilcoxon_stat = float(parts[9])
                p_value = float(parts[10])
                significance = parts[11] == 'True'
                
                data_rows.append({
                    'corruption': corruption,
                    'metric': metric,
                    'median_baseline': median_baseline,
                    'median_trained': median_trained,
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


def create_comparison_bar_chart(df, output_dir, validation_type='Internal'):
    """
    Create individual bar charts for each corruption comparing baseline vs trained model performance.
    Saves each corruption as a separate file.
    
    X-axis: Metric Type (Accuracy, Precision, Recall, Jaccard)
    Y-axis: Median Performance (%)
    Groups: Clean-trained (baseline) vs Corruption-trained for each corruption type
    """
    # Define corruption order and colors
    corruption_order = ['gaussian_noise', 'motion_blur', 'defocus_blur',
                       'smoke_effect', 'uneven_illumination', 'random']
    
    # Filter to only include corruptions that exist in data
    available_corruptions = [c for c in corruption_order if c in df['corruption'].unique()]
    
    # Metrics order
    metrics = ['accuracy', 'precision', 'recall', 'jaccard']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'Jaccard']
    
    # Define colors for baseline and trained
    colors = {'baseline': '#3498db', 'trained': '#e74c3c'}
    
    # Create individual bar chart for each corruption
    for corruption in available_corruptions:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        corruption_data = df[df['corruption'] == corruption]
        
        baseline_values = []
        trained_values = []
        p_values = []
        
        for metric in metrics:
            metric_data = corruption_data[corruption_data['metric'] == metric]
            if not metric_data.empty:
                row = metric_data.iloc[0]
                baseline_values.append(row['median_baseline'])
                trained_values.append(row['median_trained'])
                p_values.append(row['p_value'])
            else:
                baseline_values.append(0)
                trained_values.append(0)
                p_values.append(1.0)
        
        # Set up bar positions
        x = np.arange(len(metrics))
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - width/2, baseline_values, width, label='Clean-Trained',
                      color=colors['baseline'], alpha=0.85, edgecolor='black', linewidth=0.8)
        bars2 = ax.bar(x + width/2, trained_values, width, label='Corruption-Trained',
                      color=colors['trained'], alpha=0.85, edgecolor='black', linewidth=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height,
                       f'{height:.1f}',
                       ha='center', va='bottom',
                       fontsize=9, fontweight='bold')
        
        # Add significance markers
        max_height = max(max(baseline_values), max(trained_values))
        for i, p_val in enumerate(p_values):
            sig_marker = get_significance_marker(p_val)
            if sig_marker:
                y_pos = max(baseline_values[i], trained_values[i]) + max_height * 0.05
                ax.text(i, y_pos, sig_marker,
                       ha='center', va='bottom',
                       fontsize=11, fontweight='bold', color='red')
        
        # Customize plot
        corruption_label = corruption.replace('_', ' ').title()
        ax.set_title(f'{corruption_label} - {validation_type} Validation',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
        ax.set_ylabel('Median Performance (%)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=11)
        ax.set_ylim(0, min(105, max_height * 1.2))
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=10)
        
        # Add footnote
        fig.text(0.5, 0.02,
                 '* p<0.05, ** p<0.01, *** p<0.001 (Wilcoxon signed-rank test)',
                 ha='center', fontsize=9, style='italic')
        
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        
        # Save individual file
        output_path = output_dir / f'{corruption}_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()


def create_single_grouped_chart(df, output_dir, validation_type='Internal'):
    """
    Create individual bar charts for each metric showing all corruptions.
    Saves each metric as a separate file.
    
    X-axis: Corruption types
    Y-axis: Median Performance (%)
    Groups: Baseline vs Trained for each metric
    """
    # Define corruption order
    corruption_order = ['gaussian_noise', 'motion_blur', 'defocus_blur',
                       'smoke_effect', 'uneven_illumination', 'random']
    
    # Filter to only include corruptions that exist in data
    available_corruptions = [c for c in corruption_order if c in df['corruption'].unique()]
    
    # Metrics
    metrics = ['accuracy', 'precision', 'recall', 'jaccard']
    
    # Define colors
    colors = {'baseline': '#3498db', 'trained': '#e74c3c'}
    
    # Create individual bar chart for each metric
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        baseline_values = []
        trained_values = []
        p_values = []
        
        for corruption in available_corruptions:
            corruption_data = df[(df['corruption'] == corruption) & (df['metric'] == metric)]
            if not corruption_data.empty:
                row = corruption_data.iloc[0]
                baseline_values.append(row['median_baseline'])
                trained_values.append(row['median_trained'])
                p_values.append(row['p_value'])
            else:
                baseline_values.append(0)
                trained_values.append(0)
                p_values.append(1.0)
        
        # Set up bar positions
        x = np.arange(len(available_corruptions))
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - width/2, baseline_values, width, label='Clean-Trained',
                      color=colors['baseline'], alpha=0.85, edgecolor='black', linewidth=0.8)
        bars2 = ax.bar(x + width/2, trained_values, width, label='Corruption-Trained',
                      color=colors['trained'], alpha=0.85, edgecolor='black', linewidth=0.8)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height,
                       f'{height:.1f}',
                       ha='center', va='bottom',
                       fontsize=9, fontweight='bold')
        
        # Add significance markers
        max_height = max(max(baseline_values), max(trained_values))
        for i, p_val in enumerate(p_values):
            sig_marker = get_significance_marker(p_val)
            if sig_marker:
                y_pos = max(baseline_values[i], trained_values[i]) + max_height * 0.05
                ax.text(i, y_pos, sig_marker,
                       ha='center', va='bottom',
                       fontsize=11, fontweight='bold', color='red')
        
        # Customize plot
        ax.set_title(f'{metric.title()} - {validation_type} Validation',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Corruption Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Median Performance (%)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace('_', ' ').title() for c in available_corruptions],
                          rotation=45, ha='right', fontsize=10)
        ax.set_ylim(0, min(105, max_height * 1.2))
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=10)
        
        # Add footnote
        fig.text(0.5, 0.02,
                 '* p<0.05, ** p<0.01, *** p<0.001 (Wilcoxon signed-rank test)',
                 ha='center', fontsize=9, style='italic')
        
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        
        # Save individual file
        output_path = output_dir / f'{metric}_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()


# Main execution
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate comparison bar graphs from statistical_log_with_clean_median.txt'
    )
    parser.add_argument('--input', '-i',
                       default='results/statistical_log_with_clean_median.txt',
                       help='Path to statistical_log_with_clean_median.txt file')
    parser.add_argument('--output_dir', '-o',
                       default='results/bar_graphs',
                       help='Output directory for graphs')
    parser.add_argument('--validation_type', '-v',
                       default='Internal',
                       choices=['Internal', 'External'],
                       help='Type of validation (Internal or External)')
    
    args = parser.parse_args()
    
    # Setup paths
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎨 Generating comparison bar graphs...")
    print("=" * 70)
    print(f"📂 Input file: {input_path}")
    print(f"📂 Output directory: {output_dir}")
    print(f"📊 Validation type: {args.validation_type}")
    print("=" * 70)
    
    # Parse the statistical log file
    df = parse_statistical_log(input_path)
    
    # Create comparison bar charts
    print("\n📊 Creating individual bar charts for each corruption...")
    create_comparison_bar_chart(df, output_dir, validation_type=args.validation_type)
    
    print("\n📊 Creating individual bar charts for each metric...")
    create_single_grouped_chart(df, output_dir, validation_type=args.validation_type)
    
    print("\n" + "=" * 70)
    print(f"✅ All graphs saved to: {output_dir}")
    print("\nGenerated files:")
    print("  📊 <corruption>_comparison.png - One file per corruption type (6 files)")
    print("  📊 <metric>_comparison.png - One file per metric (4 files)")
    print("  📊 Total: 10 individual bar chart files")
