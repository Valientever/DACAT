#!/usr/bin/env python3
"""
analyze_corruptions.py

Given a CSV of per-video scores under clean + K corruptions,
runs:
  1) Wilcoxon signed-rank test (clean vs each corruption)
  2) Friedman test across all conditions
  3) Dunn’s post-hoc (clean vs each corruption, Holm-corrected)

Outputs all p-values and flags significance at alpha=0.05.
"""
import argparse
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare
import scikit_posthocs as sp

def parse_args():
    p = argparse.ArgumentParser(
        description="Statistical significance of model drop under corruptions")
    p.add_argument(
        "-i", "--input-csv",
        required=True,
        help="CSV file with columns: video_id,condition,score")
    p.add_argument(
        "-a", "--alpha",
        type=float,
        default=0.05,
        help="Significance level (default 0.05)")
    return p.parse_args()

def load_data(path):
    df = pd.read_csv(path)
    # basic sanity
    assert set(df.columns) >= {"video_id", "condition", "score"}, \
        "CSV must have columns: video_id,condition,score"
    return df

def wilcoxon_tests(df, alpha):
    videos = df.video_id.unique()
    clean = df[df.condition == "clean"].set_index("video_id")["score"]
    results = []
    for cond in sorted(df.condition.unique()):
        if cond == "clean": continue
        corr = df[df.condition == cond].set_index("video_id")["score"]
        # align on same videos
        clean_aligned = clean.loc[videos]
        corr_aligned = corr.loc[videos]
        stat, p = wilcoxon(clean_aligned, corr_aligned)
        results.append((cond, stat, p, p < alpha))
    return pd.DataFrame(
        results,
        columns=["corruption","wilcoxon_stat","p_value","significant"]
    )

def friedman_test(df):
    # pivot to wide: one row per video_id, one col per condition
    wide = df.pivot(index="video_id", columns="condition", values="score")
    # ensure clean is first
    cols = ["clean"] + [c for c in wide.columns if c != "clean"]
    wide = wide[cols]
    stat, p = friedmanchisquare(*[wide[c] for c in cols])
    return stat, p, cols, wide.values

def dunn_posthoc(data_matrix, cols, alpha):
    # data_matrix shape: (n_videos, n_conditions)
    pvals = sp.posthoc_dunn(data_matrix, p_adjust="holm")
    pvals.index = pvals.columns = cols
    # extract clean vs each corr
    clean_vs = pvals.loc["clean", cols[1:]].reset_index()
    clean_vs.columns = ["corruption","p_value"]
    clean_vs["significant"] = clean_vs.p_value < alpha
    return clean_vs

def main():
    args = parse_args()
    df = load_data(args.input_csv)

    print("\n=== Wilcoxon signed‑rank tests (clean vs each corruption) ===")
    wdf = wilcoxon_tests(df, args.alpha)
    print(wdf.to_string(index=False, float_format="%.4f"))

    print("\n=== Friedman omnibus test across all conditions ===")
    chi2, p_fried, cols, data_mat = friedman_test(df)
    print(f"Friedman χ² = {chi2:.3f}, p = {p_fried:.4f} "
          f"({'significant' if p_fried < args.alpha else 'n.s.'})")

    if p_fried < args.alpha:
        print("\n=== Dunn’s post‑hoc (clean vs each corruption, Holm-corrected) ===")
        dout = dunn_posthoc(data_mat, cols, args.alpha)
        print(dout.to_string(index=False, float_format="%.4f"))
    else:
        print("Skipping post‑hoc since omnibus was not significant.")

if __name__ == "__main__":
    main()
