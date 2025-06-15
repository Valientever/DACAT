#!/usr/bin/env python3
"""
generate_cross_corruption_experiments.py

This script generates the commands needed to run all cross-corruption experiments
to get p-values for every training-evaluation combination.

Current status: You have diagonal comparisons only (6 out of 49 combinations)
Missing: 43 cross-corruption experiments

This will generate commands for:
1. Clean baseline experiments
2. Cross-corruption experiments (train on X, evaluate on Y)
3. Data collection and analysis
"""

import itertools
import os
from pathlib import Path

def generate_experiment_commands():
    """Generate all necessary experiment commands for cross-corruption analysis"""
    
    # Define corruption types
    corruptions = [
        'gaussian_noise',
        'motion_blur', 
        'defocus_blur',
        'uneven_illumination',
        'smoke_effect',
        'random'
    ]
    
    # Add clean baseline
    all_conditions = ['clean'] + corruptions
    
    print("🚀 CROSS-CORRUPTION EXPERIMENT GENERATOR")
    print("=" * 60)
    print(f"Corruption types: {len(corruptions)}")
    print(f"Total conditions: {len(all_conditions)}")
    print(f"Total combinations: {len(all_conditions)} × {len(all_conditions)} = {len(all_conditions)**2}")
    print(f"Current available: 6 (diagonal only)")
    print(f"Missing experiments: {len(all_conditions)**2 - 6}")
    
    # Generate experiment matrix
    experiments_needed = []
    
    for train_corruption in all_conditions:
        for eval_corruption in all_conditions:
            # Check if this combination already exists
            if train_corruption == eval_corruption and train_corruption != 'clean':
                status = "✅ AVAILABLE"
            elif train_corruption == 'clean' and eval_corruption == 'clean':
                status = "✅ BASELINE"
            else:
                status = "❌ MISSING"
                experiments_needed.append((train_corruption, eval_corruption))
            
            print(f"Train: {train_corruption:20} → Eval: {eval_corruption:20} | {status}")
    
    print(f"\n📊 SUMMARY:")
    print(f"Experiments needed: {len(experiments_needed)}")
    
    return experiments_needed, corruptions, all_conditions

def generate_training_commands(experiments_needed):
    """Generate bash commands for running the missing experiments"""
    
    print(f"\n🔧 GENERATED COMMANDS FOR MISSING EXPERIMENTS:")
    print("=" * 80)
    
    # Create output script
    script_content = ["#!/bin/bash", "# Cross-corruption experiment commands", ""]
    
    for i, (train_corruption, eval_corruption) in enumerate(experiments_needed, 1):
        experiment_name = f"cross_{train_corruption}_to_{eval_corruption}"
        
        print(f"\n{i}. Train on '{train_corruption}' → Evaluate on '{eval_corruption}'")
        print(f"   Experiment name: {experiment_name}")
        
        # Generate training command
        if train_corruption == 'clean':
            train_flag = ""
            print(f"   Training: No corruption (clean baseline)")
        else:
            train_flag = f"--corruption {train_corruption}"
            print(f"   Training: With {train_corruption} corruption")
        
        if eval_corruption == 'clean':
            eval_flag = ""
            print(f"   Evaluation: No corruption (clean data)")
        else:
            eval_flag = f"--corruption {eval_corruption}"
            print(f"   Evaluation: With {eval_corruption} corruption")
        
        # Add commands to script
        script_content.extend([
            f"# Experiment {i}: {train_corruption} → {eval_corruption}",
            f"echo 'Starting experiment {i}: {train_corruption} → {eval_corruption}'",
            f"export EXPERIMENT_NAME='{experiment_name}'",
            f"export TRAIN_CORRUPTION='{train_corruption if train_corruption != 'clean' else ''}'",
            f"export EVAL_CORRUPTION='{eval_corruption if eval_corruption != 'clean' else ''}'",
            "",
            "# Step 1: Training",
            f"python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 10 {train_flag}",
            "",
            "# Step 2: Long-short training", 
            f"python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 10 {train_flag}",
            "",
            "# Step 3: Generate predictions",
            f"python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 {eval_flag}",
            "",
            "# Step 4: Evaluation",
            f"cd /home/santhi/Documents/DACAT/src/Cholec80",
            f"python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts",
            f"cd /home/santhi/Documents/DACAT/src/Cholec80/train_scripts",
            "",
            f"echo 'Completed experiment {i}: {train_corruption} → {eval_corruption}'",
            "sleep 10  # Brief pause between experiments",
            ""
        ])
    
    # Save the script
    script_path = "/home/santhi/Documents/DACAT/src/Cholec80/run_cross_corruption_experiments.sh"
    with open(script_path, 'w') as f:
        f.write('\n'.join(script_content))
    
    os.chmod(script_path, 0o755)  # Make executable
    
    print(f"\n💾 Generated experiment script: {script_path}")
    print(f"📏 Total commands: {len(experiments_needed)} experiments")
    
    return script_path

def generate_priority_experiments():
    """Generate a priority list of most important cross-corruption experiments"""
    
    print(f"\n🎯 PRIORITY EXPERIMENTS (Most Important First):")
    print("=" * 60)
    
    # Define high-priority combinations
    priority_combinations = [
        # Clean baseline evaluations
        ('clean', 'gaussian_noise'),
        ('clean', 'motion_blur'),
        ('clean', 'defocus_blur'),
        ('clean', 'smoke_effect'),
        
        # Cross-corruption between similar types
        ('gaussian_noise', 'motion_blur'),
        ('motion_blur', 'defocus_blur'),
        ('defocus_blur', 'uneven_illumination'),
        
        # Robustness tests (corruption → clean)
        ('gaussian_noise', 'clean'),
        ('motion_blur', 'clean'),
        ('defocus_blur', 'clean'),
    ]
    
    priority_script = ["#!/bin/bash", "# Priority cross-corruption experiments", ""]
    
    for i, (train_corruption, eval_corruption) in enumerate(priority_combinations, 1):
        experiment_name = f"priority_{train_corruption}_to_{eval_corruption}"
        
        train_flag = f"--corruption {train_corruption}" if train_corruption != 'clean' else ""
        eval_flag = f"--corruption {eval_corruption}" if eval_corruption != 'clean' else ""
        
        print(f"{i:2d}. Train: {train_corruption:15} → Eval: {eval_corruption:15} | {experiment_name}")
        
        priority_script.extend([
            f"# Priority Experiment {i}: {train_corruption} → {eval_corruption}",
            f"echo 'Starting priority experiment {i}: {train_corruption} → {eval_corruption}'",
            f"export EXPERIMENT_NAME='{experiment_name}'",
            "",
            f"python3 train.py phase --split cuhk --backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 1e-4 --random_seed --trial_name Step1 --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step 1 --epochs 5 {train_flag}",
            f"python3 train_longshort.py phase --split cuhk --backbone convnextv2 --workers 4 --seq_len 64 --lr 1e-5 --random_seed --trial_name DACAT --experiment_name $EXPERIMENT_NAME --step_1 phase_1 --step_2 phase_2 --step 2 --epochs 5 {train_flag}",
            f"python3 save_predictions_onlinev2_longshort.py phase --split cuhk --backbone convnextv2 --seq_len 1 --resume 1 --experiment_name $EXPERIMENT_NAME --step_1 phase_2 --step_3 predicts --step 3 {eval_flag}",
            f"cd /home/santhi/Documents/DACAT/src/Cholec80 && python3 eval.py --experiment_name $EXPERIMENT_NAME --predict_name predicts && cd train_scripts",
            ""
        ])
    
    # Save priority script
    priority_path = "/home/santhi/Documents/DACAT/src/Cholec80/run_priority_cross_corruption.sh"
    with open(priority_path, 'w') as f:
        f.write('\n'.join(priority_script))
    
    os.chmod(priority_path, 0o755)
    
    print(f"\n💾 Priority experiment script: {priority_path}")
    print(f"📏 Priority experiments: {len(priority_combinations)}")
    
    return priority_path

def generate_data_collection_script():
    """Generate script to collect results from all cross-corruption experiments"""
    
    collection_script = [
        "#!/usr/bin/env python3",
        '"""',
        "collect_cross_corruption_results.py",
        "",
        "Collect results from all cross-corruption experiments and generate",
        "the complete statistical analysis with p-values for all combinations.",
        '"""',
        "",
        "import os",
        "import pandas as pd",
        "import subprocess",
        "from pathlib import Path",
        "",
        "def collect_all_experiment_results():",
        '    """Collect results from all cross-corruption experiments"""',
        "",
        "    results_dir = Path('/home/santhi/Documents/DACAT/src/Cholec80/results')",
        "    all_results = []",
        "",
        "    # Find all cross-corruption experiment folders",
        "    for folder in results_dir.iterdir():",
        "        if folder.is_dir() and ('cross_' in folder.name or 'priority_' in folder.name):",
        "            print(f'Processing: {folder.name}')",
        "",
        "            # Extract train and eval conditions from folder name",
        "            if 'cross_' in folder.name:",
        "                parts = folder.name.replace('cross_', '').split('_to_')",
        "            else:",
        "                parts = folder.name.replace('priority_', '').split('_to_')",
        "",
        "            if len(parts) == 2:",
        "                train_condition, eval_condition = parts",
        "                # TODO: Extract metrics from eval_results.txt",
        "                # TODO: Add to all_results list",
        "",
        "    # Generate complete CSV",
        "    df = pd.DataFrame(all_results)",
        "    df.to_csv(results_dir / 'complete_cross_corruption_metrics.csv', index=False)",
        "    print(f'Complete results saved to: {results_dir / \"complete_cross_corruption_metrics.csv\"}')",
        "",
        "    # Run statistical analysis",
        "    cmd = [",
        "        'python3', 'analyze_test.py',",
        "        '--input', str(results_dir / 'complete_cross_corruption_metrics.csv'),",
        "        '--log', str(results_dir / 'complete_statistical_log.txt')",
        "    ]",
        "    subprocess.run(cmd)",
        "",
        "    # Generate complete heatmaps",
        "    cmd = [",
        "        'python3', 'generate_cross_corruption_heatmaps.py',",
        "        '--input', str(results_dir / 'complete_statistical_log.txt'),",
        "        '--output_dir', str(results_dir / 'complete_cross_corruption_analysis')",
        "    ]",
        "    subprocess.run(cmd)",
        "",
        "if __name__ == '__main__':",
        "    collect_all_experiment_results()",
    ]
    
    collection_path = "/home/santhi/Documents/DACAT/src/Cholec80/collect_cross_corruption_results.py"
    with open(collection_path, 'w') as f:
        f.write('\n'.join(collection_script))
    
    os.chmod(collection_path, 0o755)
    
    return collection_path

def main():
    """Main function to generate all cross-corruption experiment commands"""
    
    # Generate experiment matrix
    experiments_needed, corruptions, all_conditions = generate_experiment_commands()
    
    # Generate training commands
    full_script = generate_training_commands(experiments_needed)
    
    # Generate priority experiments
    priority_script = generate_priority_experiments()
    
    # Generate data collection script
    collection_script = generate_data_collection_script()
    
    print(f"\n🎉 CROSS-CORRUPTION EXPERIMENT SETUP COMPLETE!")
    print("=" * 80)
    print(f"📁 Generated files:")
    print(f"   1. {full_script}")
    print(f"   2. {priority_script}")
    print(f"   3. {collection_script}")
    
    print(f"\n🚀 RECOMMENDED WORKFLOW:")
    print(f"1. Start with priority experiments (faster, most important):")
    print(f"   bash {priority_script}")
    print(f"")
    print(f"2. If needed, run all experiments (will take a long time):")
    print(f"   bash {full_script}")
    print(f"")
    print(f"3. Collect results and generate complete analysis:")
    print(f"   python3 {collection_script}")
    print(f"")
    print(f"⚠️  ESTIMATED TIME:")
    print(f"   Priority experiments: ~10-15 hours")
    print(f"   All experiments: ~40-50 hours")
    print(f"   (Depending on your GPU and training speed)")

if __name__ == '__main__':
    main()
