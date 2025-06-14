#!/usr/bin/env python3
"""
Automated experiment runner for surgical phase detection with corruptions.
Runs baseline + all corruption types with multiple trials for statistical significance.
"""

import os
import subprocess
import time
from datetime import datetime

# Configuration
CORRUPTIONS = [
    ("baseline", None),
    ("gaussian_noise", "gaussian_noise"),
    ("motion_blur", "motion_blur"), 
    ("defocus_blur", "defocus_blur"),
    ("uneven_illumination", "uneven_illumination"),
    ("smoke_effect", "smoke_effect"),
    ("random", "random")
]

NUM_TRIALS = 5  # For statistical significance
BASE_DIR = "/home/santhi/Documents/DACAT/src/Cholec80"

def run_experiment(experiment_name, corruption_name=None):
    """Run a single experiment with given parameters."""
    print(f"\n{'='*50}")
    print(f"Starting: {experiment_name}")
    print(f"Corruption: {corruption_name or 'None (Clean)'}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")
    
    # Modify combine_train.sh with current experiment settings
    script_path = os.path.join(BASE_DIR, "combine_train.sh")
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Update experiment name
    content = content.replace(
        'EXPERIMENT_NAME="gn_run_5"',
        f'EXPERIMENT_NAME="{experiment_name}"'
    )
    
    # Update corruption
    if corruption_name:
        content = content.replace(
            'CORRUPTION_NAME="gaussian_noise"',
            f'CORRUPTION_NAME="{corruption_name}"'
        )
        # Ensure corruption flags are present
        content = content.replace(
            '--epochs 10  #300',
            f'--epochs 10 --corruption $CORRUPTION_NAME #300'
        )
        content = content.replace(
            '--epochs 10 #30',
            f'--epochs 10 --corruption $CORRUPTION_NAME #30'
        )
    else:
        # Remove corruption flags for baseline
        content = content.replace(
            '--corruption $CORRUPTION_NAME',
            ''
        )
    
    with open(script_path, 'w') as f:
        f.write(content)
    
    # Run the experiment
    try:
        result = subprocess.run(
            ["bash", script_path],
            cwd=BASE_DIR,
            capture_output=False,
            text=True,
            timeout=7200  # 2 hour timeout per experiment
        )
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: {experiment_name}")
            return True
        else:
            print(f"❌ FAILED: {experiment_name} (Exit code: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {experiment_name}")
        return False
    except Exception as e:
        print(f"💥 ERROR: {experiment_name} - {str(e)}")
        return False

def main():
    """Run all experiments with multiple trials."""
    start_time = time.time()
    results = {"success": [], "failed": []}
    
    print("🚀 Starting Automated Experiment Suite")
    print(f"Total experiments: {len(CORRUPTIONS) * NUM_TRIALS}")
    
    for trial in range(1, NUM_TRIALS + 1):
        for corruption_type, corruption_name in CORRUPTIONS:
            experiment_name = f"{corruption_type}_run_{trial}"
            
            success = run_experiment(experiment_name, corruption_name)
            
            if success:
                results["success"].append(experiment_name)
            else:
                results["failed"].append(experiment_name)
            
            # Brief pause between experiments
            time.sleep(10)
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("🏁 EXPERIMENT SUITE COMPLETED")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Successful: {len(results['success'])}")
    print(f"Failed: {len(results['failed'])}")
    
    if results["failed"]:
        print("\n❌ Failed experiments:")
        for exp in results["failed"]:
            print(f"  - {exp}")
    
    print(f"\n✅ Results saved in: {BASE_DIR}/results/")
    print("📊 Run statistical analysis next!")

if __name__ == "__main__":
    main()
