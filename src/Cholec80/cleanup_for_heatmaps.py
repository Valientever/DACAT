#!/usr/bin/env python3
"""
cleanup_for_heatmaps.py

This script removes unnecessary files and keeps only the essential files needed for:
1. P-value heatmap generation
2. Performance/effect size heatmap generation

KEEPS:
- statistical_log.txt (main input for heatmaps)
- all_per_video_metrics.csv (input for analysis)
- Final heatmap visualization folders
- Core analysis scripts
- normality_test_results.txt

REMOVES:
- Temporary experiment folders (run_1, run_2, etc.)
- Individual corruption experiment folders
- Check/debug folders (check, check_db, etc.)
- Intermediate PNG files in main results folder
- Temporary files and folders
"""

import os
import shutil
from pathlib import Path
import argparse

def get_files_to_keep():
    """Define which files/folders are essential for heatmap generation"""
    
    essential_files = {
        # Core input files
        'statistical_log.txt',
        'statistical_log_2.txt', 
        'all_per_video_metrics.csv',
        'normality_test_results.txt',
        
        # Final visualization folders
        'visualizations/',
        'cross_corruption_analysis/',
        'cross_corruption_both_types/',
        
        # Any baseline/clean experiment (needed for comparison)
        'baseline/',
        'base/',
    }
    
    return essential_files

def get_folders_to_remove():
    """Define which folders can be safely removed"""
    
    removable_patterns = [
        # Temporary experiment runs
        'run_1/', 'run_2/', 'run_3/', 'run_4/', 'run_5/',
        'gn_run_1/', 'gn_run_2/', 'gn_run_3/', 'gn_run_4/', 'gn_run_5/',
        'mb_run_1/', 'mb_run_2/', 'mb_run_3/', 'mb_run_4/',
        
        # Individual corruption folders (data is now in CSV)
        'gaussian_noise/', 'motion_blur/', 'defocus_blur/', 'smoke_effect/',
        'uneven_illumination/', 'random_corruptions/',
        'gaussian_noise_w28/', 'motion_blur_w28/', 'defocus_blur_w28/',
        'smoke_effect_w28/', 'uneven_illumination_w28/', 'random_corruptions_w28/',
        'random_w28/',
        
        # Single corruption runs
        'noise_1/', 'motion_1/', 'defocus_1/', 'smoke_1/', 'u_ill_1/',
        
        # Debug/check folders
        'check/', 'check_db/', 'check_mb/', 'check_se/', 'check_ui/',
        
        # Alien/test folders
        'alien_1/', 'alien_nw1/', 'alien_all_time_nw30/', 'alien_all_time_nw30_t2/',
        
        # Epoch test folders
        '1_epoch_w28/', '10_epoch_w28/', '11_epoch_w28/',
        
        # Other test folders
        'mickey/', 'testing_path/', 'train/', 'phase_1/',
        'baseline_base/', 'Smoke/',
    ]
    
    return removable_patterns

def get_files_to_remove():
    """Define which individual files can be removed"""
    
    removable_files = [
        # Individual PNG files in results root (now in organized folders)
        'combined_analysis.png',
        'effect_size_heatmap.png', 
        'pvalue_heatmap.png',
        'significance_matrix.png',
        'summary_statistics.txt',
        'check.png',
        
        # Config files that aren't needed for heatmaps
        'ckpts.yaml',
    ]
    
    return removable_files

def cleanup_results_folder(results_path, dry_run=True):
    """Clean up the results folder"""
    
    results_path = Path(results_path)
    
    if not results_path.exists():
        print(f"❌ Results folder not found: {results_path}")
        return
    
    print(f"🧹 Cleaning up results folder: {results_path}")
    print(f"🔍 Mode: {'DRY RUN (preview only)' if dry_run else 'ACTUAL CLEANUP'}")
    print("=" * 60)
    
    folders_to_remove = get_folders_to_remove()
    files_to_remove = get_files_to_remove()
    
    removed_count = 0
    saved_space = 0
    
    # Remove folders
    print("\n📁 FOLDERS TO REMOVE:")
    for item in results_path.iterdir():
        if item.is_dir():
            folder_name = item.name + '/'
            if folder_name in folders_to_remove:
                # Calculate size
                try:
                    folder_size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                    saved_space += folder_size
                    size_mb = folder_size / (1024 * 1024)
                    
                    print(f"  🗑️  {folder_name} ({size_mb:.1f} MB)")
                    
                    if not dry_run:
                        shutil.rmtree(item)
                    removed_count += 1
                    
                except Exception as e:
                    print(f"  ❌ Error processing {folder_name}: {e}")
    
    # Remove files
    print("\n📄 FILES TO REMOVE:")
    for item in results_path.iterdir():
        if item.is_file() and item.name in files_to_remove:
            try:
                file_size = item.stat().st_size
                saved_space += file_size
                size_kb = file_size / 1024
                
                print(f"  🗑️  {item.name} ({size_kb:.1f} KB)")
                
                if not dry_run:
                    item.unlink()
                removed_count += 1
                
            except Exception as e:
                print(f"  ❌ Error processing {item.name}: {e}")
    
    # Show what's being kept
    print("\n✅ ESSENTIAL FILES/FOLDERS KEPT:")
    essential_files = get_files_to_keep()
    for item in results_path.iterdir():
        item_name = item.name + ('/' if item.is_dir() else '')
        if item_name in essential_files or any(item.name.startswith(ef.rstrip('/')) for ef in essential_files):
            print(f"  📌 {item_name}")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 CLEANUP SUMMARY:")
    print(f"  Items to remove: {removed_count}")
    print(f"  Space to save: {saved_space / (1024 * 1024):.1f} MB")
    
    if dry_run:
        print(f"\n🔄 To perform actual cleanup, run:")
        print(f"   python3 cleanup_for_heatmaps.py --execute")
    else:
        print(f"\n✅ Cleanup completed!")

def main():
    parser = argparse.ArgumentParser(description="Clean up results folder for heatmap generation")
    parser.add_argument('--results_path', 
                       default='/home/santhi/Documents/DACAT/src/Cholec80/results',
                       help='Path to results folder')
    parser.add_argument('--execute', action='store_true',
                       help='Actually perform cleanup (default is dry run)')
    
    args = parser.parse_args()
    
    cleanup_results_folder(args.results_path, dry_run=not args.execute)

if __name__ == '__main__':
    main()
