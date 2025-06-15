#!/usr/bin/env python3
"""
Current Status Summary for Cross-Corruption Analysis
====================================================

This script provides a comprehensive overview of the current state 
of the cross-corruption analysis project.
"""

import os
from pathlib import Path
import datetime

def print_status():
    print("🔍 CROSS-CORRUPTION ANALYSIS - CURRENT STATUS")
    print("=" * 70)
    print(f"📅 Status check performed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check existing data
    print("📊 CURRENT DATA AVAILABLE:")
    print("-" * 30)
    statistical_log = Path("/home/santhi/Documents/DACAT/src/Cholec80/results/statistical_log.txt")
    if statistical_log.exists():
        with open(statistical_log, 'r') as f:
            content = f.read()
            corruption_types = []
            for line in content.split('\n'):
                if line.startswith('=== Corruption:'):
                    corruption_type = line.split('Corruption: ')[1].split(' ===')[0]
                    corruption_types.append(corruption_type)
        
        print(f"✅ Diagonal experiments completed: {len(corruption_types)}")
        print(f"   Available corruptions: {', '.join(corruption_types)}")
        print("   ⚠️  Note: These are same-corruption experiments (train X → eval X)")
    else:
        print("❌ No statistical_log.txt found")
    
    print()
    
    # Check generated scripts
    print("🛠️  EXPERIMENT SCRIPTS READY:")
    print("-" * 30)
    
    priority_script = Path("/home/santhi/Documents/DACAT/src/Cholec80/run_priority_cross_corruption.sh")
    full_script = Path("/home/santhi/Documents/DACAT/src/Cholec80/run_cross_corruption_experiments.sh")
    
    if priority_script.exists():
        print("✅ Priority experiment script: run_priority_cross_corruption.sh")
        print("   Contains: 10 most important cross-corruption combinations")
    else:
        print("❌ Priority script not found")
        
    if full_script.exists():
        print("✅ Full experiment script: run_cross_corruption_experiments.sh")  
        print("   Contains: All 42 missing cross-corruption combinations")
    else:
        print("❌ Full script not found")
    
    print()
    
    # Check visualizations
    print("📈 CURRENT VISUALIZATIONS:")
    print("-" * 30)
    viz_dir = Path("/home/santhi/Documents/DACAT/src/Cholec80/results/visualizations")
    if viz_dir.exists():
        png_files = list(viz_dir.glob("*.png"))
        if png_files:
            print("✅ Heatmaps generated:")
            for png_file in png_files:
                print(f"   - {png_file.name}")
            print("   ⚠️  Note: Current heatmaps show only diagonal data (incomplete matrix)")
        else:
            print("❌ No visualization files found")
    else:
        print("❌ Visualizations directory not found")
    
    print()
    
    # Recommendations
    print("🎯 RECOMMENDED NEXT STEPS:")
    print("-" * 30)
    print("1. 🏃‍♂️ START WITH PRIORITY EXPERIMENTS (recommended)")
    print("   Command: cd /home/santhi/Documents/DACAT/src/Cholec80 && bash run_priority_cross_corruption.sh")
    print("   Time: ~10-15 hours")
    print("   Result: Most important cross-corruption data for analysis")
    print()
    
    print("2. 🔬 OR RUN FULL EXPERIMENT SET (comprehensive)")
    print("   Command: cd /home/santhi/Documents/DACAT/src/Cholec80 && bash run_cross_corruption_experiments.sh")
    print("   Time: ~40-50 hours")
    print("   Result: Complete 7×7 cross-corruption matrix")
    print()
    
    print("3. 📊 AFTER EXPERIMENTS: Collect results and regenerate analysis")
    print("   Command: python3 collect_cross_corruption_results.py")
    print("   Command: python3 generate_cross_corruption_heatmaps.py")
    print()
    
    print("💡 CURRENT LIMITATIONS:")
    print("-" * 30)
    print("• Heatmaps show only diagonal (6/49 combinations)")
    print("• Cross-corruption effects not yet measured")
    print("• Statistical significance limited to same-corruption comparisons")
    print("• Cannot answer questions like: 'How does gaussian_noise training affect motion_blur evaluation?'")
    print()
    
    print("🎉 AFTER COMPLETION, YOU'LL HAVE:")
    print("-" * 30)
    print("• Complete 7×7 p-value heatmap showing all cross-corruption effects")
    print("• Effect size heatmap quantifying the magnitude of differences")
    print("• Statistical significance matrix for all combinations")
    print("• Data to answer research questions about corruption robustness and transfer")

if __name__ == "__main__":
    print_status()
