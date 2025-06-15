#!/usr/bin/env python3
"""
Quick script to generate cross-corruption heatmaps from statistical_log_2.txt
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Use statistical_log_2.txt that you have open
    input_file = "results/statistical_log_2.txt"
    output_dir = "results/cross_corruption_analysis_2"
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"❌ Input file not found: {input_file}")
        print("Available statistical log files:")
        results_dir = Path("results")
        if results_dir.exists():
            for f in results_dir.glob("statistical_log*.txt"):
                print(f"  - {f}")
        return
    
    # Run the cross-corruption analysis
    cmd = [
        "python3", "generate_cross_corruption_heatmaps.py",
        "--input", input_file,
        "--output_dir", output_dir
    ]
    
    print(f"🚀 Running cross-corruption analysis on {input_file}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Analysis completed successfully!")
        print(result.stdout)
    else:
        print("❌ Analysis failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

if __name__ == "__main__":
    main()
