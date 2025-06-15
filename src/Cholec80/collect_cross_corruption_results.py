#!/usr/bin/env python3
"""
collect_cross_corruption_results.py

Collect results from all cross-corruption experiments and generate
the complete statistical analysis with p-values for all combinations.
"""

import os
import pandas as pd
import subprocess
from pathlib import Path

def collect_all_experiment_results():
    """Collect results from all cross-corruption experiments"""

    results_dir = Path('/home/santhi/Documents/DACAT/src/Cholec80/results')
    all_results = []

    # Find all cross-corruption experiment folders
    for folder in results_dir.iterdir():
        if folder.is_dir() and ('cross_' in folder.name or 'priority_' in folder.name):
            print(f'Processing: {folder.name}')

            # Extract train and eval conditions from folder name
            if 'cross_' in folder.name:
                parts = folder.name.replace('cross_', '').split('_to_')
            else:
                parts = folder.name.replace('priority_', '').split('_to_')

            if len(parts) == 2:
                train_condition, eval_condition = parts
                # TODO: Extract metrics from eval_results.txt
                # TODO: Add to all_results list

    # Generate complete CSV
    df = pd.DataFrame(all_results)
    df.to_csv(results_dir / 'complete_cross_corruption_metrics.csv', index=False)
    print(f'Complete results saved to: {results_dir / "complete_cross_corruption_metrics.csv"}')

    # Run statistical analysis
    cmd = [
        'python3', 'analyze_test.py',
        '--input', str(results_dir / 'complete_cross_corruption_metrics.csv'),
        '--log', str(results_dir / 'complete_statistical_log.txt')
    ]
    subprocess.run(cmd)

    # Generate complete heatmaps
    cmd = [
        'python3', 'generate_cross_corruption_heatmaps.py',
        '--input', str(results_dir / 'complete_statistical_log.txt'),
        '--output_dir', str(results_dir / 'complete_cross_corruption_analysis')
    ]
    subprocess.run(cmd)

if __name__ == '__main__':
    collect_all_experiment_results()