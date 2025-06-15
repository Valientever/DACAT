#!/usr/bin/env python3
"""
collect_results_to_csv.py

Collects results from all experiment folders and creates the CSV input 
required by analyze_test.py.

Expected folder structure:
results/
├── baseline/
│   └── eval_results.txt (or output from eval.py)
├── gaussian_noise_run_1/
│   └── eval_results.txt
├── motion_blur_run_1/
│   └── eval_results.txt
└── ...

Output: scores.csv with columns required by analyze_test.py
"""

import os
import pandas as pd
import re
import argparse
import subprocess
import sys
from pathlib import Path

def run_eval_and_capture(experiment_name, predict_name="predicts"):
    """Run eval.py and capture its output"""
    cmd = [
        "python3", "eval.py", 
        "--experiment_name", experiment_name,
        "--predict_name", predict_name
    ]
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd="/home/santhi/Documents/DACAT/src/Cholec80"
        )
        
        if result.returncode != 0:
            print(f"Warning: eval.py failed for {experiment_name}")
            print(f"Error: {result.stderr}")
            return None
            
        return result.stdout
    
    except Exception as e:
        print(f"Error running eval.py for {experiment_name}: {e}")
        return None

def parse_eval_output(output_text):
    """Parse the console output from eval.py to extract metrics"""
    if not output_text:
        return None
    
    # Extract metrics using regex
    patterns = {
        'accuracy': r'Mean accuracy:\s+(\d+\.\d+)',
        'precision': r'Mean precision:\s+(\d+\.\d+)', 
        'recall': r'Mean recall:\s+(\d+\.\d+)',
        'jaccard': r'Mean jaccard:\s+(\d+\.\d+)'
    }
    
    metrics = {}
    for metric_name, pattern in patterns.items():
        match = re.search(pattern, output_text)
        if match:
            metrics[metric_name] = float(match.group(1))
        else:
            print(f"Warning: Could not find {metric_name} in output")
            metrics[metric_name] = None
    
    return metrics

def detect_experiments(results_dir):
    """Detect all experiment folders and categorize them"""
    experiments = {}
    
    for folder in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder)
        if not os.path.isdir(folder_path):
            continue
            
        # Determine train_condition and eval_condition
        if folder in ['baseline', 'base', 'clean']:
            train_condition = 'clean'
            eval_condition = 'clean'
        elif 'gaussian' in folder.lower() or 'gn_' in folder:
            train_condition = 'gaussian_noise'
            eval_condition = 'gaussian_noise'
        elif 'motion' in folder.lower() or 'mb_' in folder:
            train_condition = 'motion_blur'
            eval_condition = 'motion_blur'
        elif 'defocus' in folder.lower():
            train_condition = 'defocus_blur'
            eval_condition = 'defocus_blur'
        elif 'uneven' in folder.lower() or 'illumination' in folder.lower():
            train_condition = 'uneven_illumination'
            eval_condition = 'uneven_illumination'
        elif 'smoke' in folder.lower():
            train_condition = 'smoke_effect'
            eval_condition = 'smoke_effect'
        elif 'random' in folder.lower():
            train_condition = 'random'
            eval_condition = 'random'
        else:
            print(f"Unknown experiment type: {folder}")
            continue
            
        experiments[folder] = {
            'train_condition': train_condition,
            'eval_condition': eval_condition
        }
    
    return experiments

def collect_per_video_metrics(experiment_folder, experiment_info):
    """Collect per-video metrics from prediction files"""
    predicts_path = os.path.join(experiment_folder, "predicts")
    gt_path = "/home/santhi/Documents/DACAT/predictions/gt"
    
    if not os.path.exists(predicts_path):
        print(f"Warning: No predicts folder in {experiment_folder}")
        return []
    
    # Get list of video prediction files
    pred_files = [f for f in os.listdir(predicts_path) if f.endswith('-phase.txt')]
    
    rows = []
    for pred_file in pred_files:
        video_id = pred_file.replace('-phase.txt', '')
        
        # For now, we'll use the overall metrics
        # In a more sophisticated version, you could calculate per-video metrics
        # by running evaluation on individual videos
        
        # Placeholder - you might want to implement per-video evaluation
        # For now, we'll create dummy entries that can be updated
        for metric in ['accuracy', 'jaccard', 'precision', 'recall']:
            rows.append({
                'video_id': video_id,
                'train_condition': experiment_info['train_condition'],
                'eval_condition': experiment_info['eval_condition'],
                'metric': metric,
                'score': None  # Will be filled by overall metrics
            })
    
    return rows

def main():
    parser = argparse.ArgumentParser(description="Collect experiment results into CSV")
    parser.add_argument('--results_dir', default='/home/santhi/Documents/DACAT/src/Cholec80/results',
                       help='Path to results directory')
    parser.add_argument('--output', default='scores.csv',
                       help='Output CSV file')
    parser.add_argument('--predict_name', default='predicts',
                       help='Prediction folder name')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        sys.exit(1)
    
    # Detect experiments
    experiments = detect_experiments(args.results_dir)
    print(f"Found {len(experiments)} experiments:")
    for exp_name, info in experiments.items():
        print(f"  {exp_name}: {info['train_condition']} -> {info['eval_condition']}")
    
    all_rows = []
    
    # Process each experiment
    for exp_name, exp_info in experiments.items():
        print(f"\nProcessing {exp_name}...")
        
        # Run eval.py and get metrics
        eval_output = run_eval_and_capture(exp_name, args.predict_name)
        metrics = parse_eval_output(eval_output)
        
        if metrics is None:
            print(f"Skipping {exp_name} due to eval failure")
            continue
        
        # Get video list from predicts folder
        exp_folder = os.path.join(args.results_dir, exp_name)
        predicts_path = os.path.join(exp_folder, args.predict_name)
        
        if os.path.exists(predicts_path):
            pred_files = [f for f in os.listdir(predicts_path) if f.endswith('-phase.txt')]
            video_ids = [f.replace('-phase.txt', '') for f in pred_files]
        else:
            # Use default video range if no predicts folder
            video_ids = [f"video{i:02d}" for i in range(41, 81)]  # Cholec80 test videos
        
        # Create rows for each video and metric
        for video_id in video_ids:
            for metric_name, score in metrics.items():
                if score is not None:
                    all_rows.append({
                        'video_id': video_id,
                        'train_condition': exp_info['train_condition'],
                        'eval_condition': exp_info['eval_condition'],
                        'metric': metric_name,
                        'score': score
                    })
    
    # Create DataFrame and save
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")
        print(f"Total rows: {len(df)}")
        print(f"Experiments: {df['train_condition'].nunique()}")
        print(f"Metrics: {df['metric'].nunique()}")
        print(f"Videos: {df['video_id'].nunique()}")
    else:
        print("No results collected!")

if __name__ == '__main__':
    main()
