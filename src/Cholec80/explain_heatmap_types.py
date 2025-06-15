#!/usr/bin/env python3
"""
CLARIFICATION: P-Values vs Effect Sizes in Cross-Corruption Heatmaps

This script explains the difference between the two types of heatmaps generated:

1. P-VALUE HEATMAPS (*_pvalue_heatmap.png):
   - Show statistical significance of differences
   - Color: Green = Low p-values (< 0.05) = Statistically significant
   - Color: Red = High p-values (≥ 0.05) = Not statistically significant
   - Arrows: ↑ = Positive effect, ↓ = Negative effect
   - Values: Actual p-values from Wilcoxon tests

2. EFFECT SIZE HEATMAPS (*_effect_size_heatmap.png):
   - Show magnitude and direction of performance differences
   - Color: Green = Positive values = Better performance 
   - Color: Red = Negative values = Worse performance
   - Asterisks (*): Mark statistically significant results
   - Values: Actual performance differences (e.g., +5.2% accuracy)

INTERPRETATION:
- P-value heatmaps tell you: "Is this difference real/reliable?"
- Effect size heatmaps tell you: "How big is this difference?"

For your research discussion:
- Use P-VALUE heatmaps to show statistical significance patterns
- Use EFFECT SIZE heatmaps to show practical significance and direction
"""

import pandas as pd

def compare_visualization_types():
    print("=" * 80)
    print("📊 CROSS-CORRUPTION HEATMAP TYPES COMPARISON")
    print("=" * 80)
    
    print("\n🔴 P-VALUE HEATMAPS:")
    print("   Purpose: Show statistical significance")
    print("   Values:  P-values from Wilcoxon signed-rank test")
    print("   Range:   0.0000 to 1.0000")
    print("   Colors:  Green = Significant (p < 0.05), Red = Not significant (p ≥ 0.05)")
    print("   Markers: ↑ = Positive effect, ↓ = Negative effect")
    print("   Files:   *_pvalue_heatmap.png")
    
    print("\n🟢 EFFECT SIZE HEATMAPS:")
    print("   Purpose: Show magnitude and direction of differences")
    print("   Values:  Performance differences (e.g., accuracy gain/loss)")
    print("   Range:   Negative (worse) to Positive (better)")
    print("   Colors:  Green = Better performance, Red = Worse performance")
    print("   Markers: * = Statistically significant (p < 0.05)")
    print("   Files:   *_effect_size_heatmap.png")
    
    print("\n📋 WHEN TO USE WHICH:")
    print("   P-value heatmaps:")
    print("   ✓ To answer: 'Which effects are statistically reliable?'")
    print("   ✓ For hypothesis testing and significance discussion")
    print("   ✓ To identify robust findings across different corruptions")
    
    print("\n   Effect size heatmaps:")
    print("   ✓ To answer: 'Which effects are practically meaningful?'")
    print("   ✓ For discussing magnitude of improvements/degradations")
    print("   ✓ To prioritize which corruption training strategies work best")
    
    print("\n🎯 FOR YOUR RESEARCH PRESENTATION:")
    print("   1. Show p-value heatmaps first to establish significance")
    print("   2. Follow with effect size heatmaps to show practical impact")
    print("   3. Combined view shows both aspects together")
    
    print("\n📁 Generated files in: results/cross_corruption_both_types/")
    print("   - Individual p-value heatmaps for each metric")
    print("   - Individual effect size heatmaps for each metric") 
    print("   - Combined p-value overview")
    print("   - Summary statistics")

if __name__ == "__main__":
    compare_visualization_types()
