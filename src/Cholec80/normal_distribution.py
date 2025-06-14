from scipy.stats import shapiro
import pandas as pd
import numpy as np

# load your aggregated CSV
df = pd.read_csv('/home/santhi/Documents/DACAT/src/Cholec80/results/all_per_video_metrics.csv')

# Open output file
output_file = '/home/santhi/Documents/DACAT/src/Cholec80/results/normality_test_results.txt'
with open(output_file, 'w') as f:
    f.write("=== NORMALITY TEST RESULTS (Shapiro-Wilk) ===\n")
    f.write("Testing all metrics differences: corruption_trained - clean_trained\n\n")
    
    dataset = ["gaussian_noise", "motion_blur", "defocus_blur", "uneven_illumination", "smoke_effect", "random_corruptions"]
    metrics = ["accuracy", "jaccard", "precision", "recall"]
    
    # Summary statistics
    normal_count = 0
    total_tests = 0
    
    for metric in metrics:
        f.write(f"{'='*60}\n")
        f.write(f"METRIC: {metric.upper()}\n")
        f.write(f"{'='*60}\n")
        print(f"\n{'='*60}")
        print(f"TESTING METRIC: {metric.upper()}")
        print(f"{'='*60}")
        
        for corruption in dataset:
            # isolate clean vs corruption for current metric
            sub = df[(df.eval_condition==corruption) & (df.metric==metric)]
            wide = sub.pivot(index='video_id', columns='train_condition', values='score')
            
            # Check if both clean and corruption columns exist
            if 'clean' not in wide.columns or corruption not in wide.columns:
                result = f"Shapiro-Wilk: {corruption} - SKIPPED (missing data)"
                print(result)
                f.write(result + "\n")
                continue
            
            diffs = wide[corruption].values - wide['clean'].values
            
            # Remove NaN values
            diffs = diffs[~pd.isna(diffs)]
            
            if len(diffs) < 3:
                result = f"Shapiro-Wilk: {corruption} - SKIPPED (insufficient data: n={len(diffs)})"
                print(result)
                f.write(result + "\n")
                continue
            
            stat, p = shapiro(diffs)
            result = f"Shapiro-Wilk: {corruption} W={stat:.3f}, p={p:.3g}"
            print(result)
            f.write(result + "\n")
            
            total_tests += 1
            
            # Interpretation
            if p > 0.05:
                interpretation = "  → NORMAL distribution (use parametric tests)"
                normal_count += 1
            else:
                interpretation = "  → NOT NORMAL distribution (use non-parametric tests)"
            print(interpretation)
            f.write(interpretation + "\n")
        
        f.write("\n")
    
    f.write(f"\n{'='*60}\n")
    f.write("SUMMARY STATISTICS\n")
    f.write(f"{'='*60}\n")
    f.write(f"Total tests performed: {total_tests}\n")
    f.write(f"Normal distributions: {normal_count}\n")
    f.write(f"Non-normal distributions: {total_tests - normal_count}\n")
    f.write(f"Normality rate: {normal_count/total_tests*100:.1f}%\n\n")
    
    f.write("=== CONCLUSION ===\n")
    if normal_count / total_tests < 0.5:
        f.write("MAJORITY of datasets are NOT normally distributed.\n")
        f.write("Recommendation: Use Wilcoxon signed-rank test (non-parametric).\n")
    else:
        f.write("MAJORITY of datasets are normally distributed.\n")
        f.write("Recommendation: Consider parametric tests (t-test) but verify assumptions.\n")
    
    print(f"\n{'='*60}")
    print("SUMMARY:")
    print(f"Total tests: {total_tests}")
    print(f"Normal: {normal_count} ({normal_count/total_tests*100:.1f}%)")
    print(f"Non-normal: {total_tests - normal_count} ({(total_tests-normal_count)/total_tests*100:.1f}%)")

print(f"\n✅ Results saved to: {output_file}")
