from scipy.stats import ttest_rel

# Replace these with your actual results
dacat = [91.08, 89.18 , 91.47, 90.95]
ours =  [90.08,
89.88,
90.24,
90.80
] 

# Paired t-test
t_stat, p_value = ttest_rel(ours, dacat)

print(f"Paired t-test statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpret the result
#If p-value < 0.05 → Your model’s improvement is statistically significant

# If p-value ≥ 0.05 → The improvement might just be due to random chance.


import numpy as np

mean_baseline = np.mean(dacat)
std_baseline = np.std(dacat)

mean_ours = np.mean(ours)
std_ours = np.std(ours)

print(f"DACAT: {mean_baseline:.2f} ± {std_baseline:.2f}")
print(f"Ours:  {mean_ours:.2f} ± {std_ours:.2f}")


