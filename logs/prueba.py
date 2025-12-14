import pandas as pd
import numpy as np
import ast
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# =========================
# CONFIG
# =========================
CSV_PATH = "results.csv"
FIG_DIR = "./figures"
sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 300

import os
os.makedirs(FIG_DIR, exist_ok=True)

# =========================
# LOAD & CLEAN
# =========================
df = pd.read_csv(CSV_PATH)

def parse_dict_column(s):
    if pd.isna(s):
        return {}
    s = re.sub(r"np\.float64\(([^)]+)\)", r"\1", s)
    return ast.literal_eval(s)

df["controller_values"] = df["controller_values"].apply(parse_dict_column)
df["controller_counts"] = df["controller_counts"].apply(parse_dict_column)
df["controller_bias"] = df["controller_bias"].apply(parse_dict_column)

print("Loaded rows:", len(df))
print("Datasets:", df["dataset"].unique())
print("Runs:", df["run_id"].nunique())

# =========================
# HELPER: FINAL STATE PER RUN
# =========================
final_rows = (
    df.sort_values("round")
      .groupby(["dataset", "run_id"])
      .tail(1)
      .reset_index(drop=True)
)

# =========================================================
# METRIC 1: CONVERGENCE CURVE (mean ± std, per dataset)
# =========================================================
conv = (
    df.groupby(["dataset", "round"])
      .agg(
          mean_best=("best_so_far", "mean"),
          std_best=("best_so_far", "std")
      )
      .reset_index()
)

for dataset in conv["dataset"].unique():
    d = conv[conv["dataset"] == dataset]

    plt.figure(figsize=(7, 4))
    plt.plot(d["round"], d["mean_best"], label="mean")
    plt.fill_between(
        d["round"],
        d["mean_best"] - d["std_best"],
        d["mean_best"] + d["std_best"],
        alpha=0.25
    )
    plt.xlabel("Iteration")
    plt.ylabel("Best F1 (macro)")
    plt.title(f"HH convergence — {dataset}")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/convergence_{dataset}.png")
    plt.close()

# =========================================================
# METRIC 2: FINAL SCORE DISTRIBUTION (per dataset)
# =========================================================
plt.figure(figsize=(7, 4))
sns.boxplot(data=final_rows, x="dataset", y="best_so_far")
plt.ylabel("Final best F1 (macro)")
plt.title("Final HH performance distribution")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/final_score_boxplot.png")
plt.close()

# =========================================================
# METRIC 3: HEURISTIC USAGE (FINAL CONTROLLER ONLY)
# =========================================================
heuristic_counter = Counter()

for counts in final_rows["controller_counts"]:
    heuristic_counter.update(counts)

heuristic_usage = (
    pd.DataFrame(
        heuristic_counter.items(),
        columns=["heuristic", "count"]
    )
    .sort_values("count", ascending=False)
)

plt.figure(figsize=(7, 4))
sns.barplot(
    data=heuristic_usage,
    y="heuristic",
    x="count"
)
plt.title("Heuristic selection frequency (final controller)")
plt.xlabel("Total selections")
plt.ylabel("")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/heuristic_usage.png")
plt.close()

# =========================================================
# METRIC 4: QUALITY vs TIME (per dataset & heuristic)
# =========================================================
tradeoff = (
    df.groupby(["dataset", "heuristic"])
      .agg(
          mean_score=("score", "mean"),
          mean_time=("round_time", "mean")
      )
      .reset_index()
)

for dataset in tradeoff["dataset"].unique():
    d = tradeoff[tradeoff["dataset"] == dataset]

    plt.figure(figsize=(6, 5))
    plt.scatter(d["mean_time"], d["mean_score"])

    for _, r in d.iterrows():
        plt.text(
            r["mean_time"],
            r["mean_score"],
            r["heuristic"],
            fontsize=9
        )

    plt.xlabel("Mean time per iteration (s)")
    plt.ylabel("Mean F1 (macro)")
    plt.title(f"Quality vs computational cost — {dataset}")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/quality_vs_time_{dataset}.png")
    plt.close()

# =========================================================
# METRIC 5: MODEL USAGE (FINAL ROUND ONLY)
# =========================================================
model_usage = (
    final_rows
    .groupby("model")
    .size()
    .reset_index(name="count")
)

plt.figure(figsize=(5, 4))
sns.barplot(data=model_usage, x="model", y="count")
plt.title("Model selection frequency (final decision)")
plt.ylabel("Runs")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/model_usage.png")
plt.close()

# =========================================================
# METRIC 6: CONTROLLER VALUE EVOLUTION (mean ± std)
# =========================================================
values_rows = []

for _, row in df.iterrows():
    for h, v in row["controller_values"].items():
        values_rows.append({
            "dataset": row["dataset"],
            "round": row["round"],
            "heuristic": h,
            "value": float(v)
        })

values_df = pd.DataFrame(values_rows)

for dataset in values_df["dataset"].unique():
    d = values_df[values_df["dataset"] == dataset]

    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=d,
        x="round",
        y="value",
        hue="heuristic",
        estimator="mean",
        ci="sd"
    )
    plt.title(f"Controller value estimates — {dataset}")
    plt.xlabel("Iteration")
    plt.ylabel("Estimated value")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/controller_values_{dataset}.png")
    plt.close()

# =========================================================
# SUMMARY TABLE (for paper)
# =========================================================
summary = (
    final_rows
    .groupby("dataset")
    .agg(
        mean_f1=("best_so_far", "mean"),
        std_f1=("best_so_far", "std")
    )
    .reset_index()
)

summary.to_csv(f"{FIG_DIR}/summary_results.csv", index=False)

print("\n=== ANALYSIS COMPLETED SUCCESSFULLY ===")
print(summary)
