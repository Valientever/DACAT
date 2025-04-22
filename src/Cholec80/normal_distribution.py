from scipy.stats import shapiro
import pandas as pd

# load your aggregated CSV
df = pd.read_csv('/home/santhi/Documents/DACAT/src/Cholec80/results/all_per_video_metrics.csv')

dataset = ["gaussian_noise", "motion_blur", "defocus_blur", "uneven_illumination", "smoke_effect", "random_corruptions"]
for corruption in dataset:
    # isolate clean vs corruption for accuracy
    sub = df[(df.eval_condition==corruption) & (df.metric=='accuracy')]
    wide = sub.pivot(index='video_id', columns='train_condition', values='score')
    diffs = wide[corruption].values - wide['clean'].values

    stat, p = shapiro(diffs)
    print(f"Shapiro-Wilk: {corruption} W={stat:.3f}, p={p:.3g}")
# # isolate clean vs gaussian_noise for accuracy
# sub = df[(df.eval_condition=='gaussian_noise') & (df.metric=='accuracy')]
# wide = sub.pivot(index='video_id', columns='train_condition', values='score')
# diffs = wide['gaussian_noise'].values - wide['clean'].values

# stat, p = shapiro(diffs)
# print(f"Shapiro-Wilk: gaussian_noise W={stat:.3f}, p={p:.3g}")
