#!/usr/bin/env python3
"""
collect_all_metrics.py

Collect all existing metrics.csv files from various cross-corruption experiments
and combine them into a single all_per_video_metrics.csv file.
"""

import os
import pandas as pd
from pathlib import Path

def main():
    results_dir = Path('/home/santhi/Documents/DACAT/src/Cholec80/results')
    
    # Find all metrics.csv files
    metrics_files = list(results_dir.glob('**/metrics.csv'))
    
    print(f"Found {len(metrics_files)} metrics.csv files")
    
    all_data = []
    successful_files = 0
    
    for metrics_file in metrics_files:
        try:
            # Read the CSV file
            df = pd.read_csv(metrics_file)
            
            # Check if it has the expected columns
            expected_cols = ['video_id', 'train_condition', 'eval_condition', 'metric', 'score']
            if not all(col in df.columns for col in expected_cols):
                print(f"⚠️  Skipping {metrics_file} - missing expected columns")
                print(f"    Columns found: {list(df.columns)}")
                continue
            
            # Add the data
            all_data.append(df)
            successful_files += 1
            
            # Print info about this file
            train_cond = df['train_condition'].iloc[0] if len(df) > 0 else 'unknown'
            eval_cond = df['eval_condition'].iloc[0] if len(df) > 0 else 'unknown'
            print(f"✅ {metrics_file.parent.parent.name}/{metrics_file.parent.name} -> {train_cond} -> {eval_cond} ({len(df)} rows)")
            
        except Exception as e:
            print(f"❌ Error reading {metrics_file}: {e}")
            continue
    
    if not all_data:
        print("❌ No valid metrics files found!")
        return
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"\n📊 Combined Statistics:")
    print(f"   Total files processed: {successful_files}")
    print(f"   Total rows: {len(combined_df)}")
    print(f"   Train conditions: {sorted(combined_df['train_condition'].unique())}")
    print(f"   Eval conditions: {sorted(combined_df['eval_condition'].unique())}")
    print(f"   Metrics: {sorted(combined_df['metric'].unique())}")
    print(f"   Video range: {combined_df['video_id'].min()}-{combined_df['video_id'].max()}")
    
    # Check for duplicates
    duplicates = combined_df.duplicated(subset=['video_id', 'train_condition', 'eval_condition', 'metric'])
    if duplicates.any():
        print(f"⚠️  Found {duplicates.sum()} duplicate entries - removing them")
        combined_df = combined_df.drop_duplicates(subset=['video_id', 'train_condition', 'eval_condition', 'metric'])
    
    # Save the combined file
    output_file = results_dir / 'all_per_video_metrics.csv'
    combined_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Successfully created: {output_file}")
    print(f"   Final dataset: {len(combined_df)} rows")
    
    # Show combination matrix
    print(f"\n📋 Train-Eval Combination Matrix:")
    pivot = combined_df.groupby(['train_condition', 'eval_condition']).size().unstack(fill_value=0)
    print(pivot)

if __name__ == '__main__':
    main()
