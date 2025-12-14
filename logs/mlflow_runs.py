import pandas as pd
from scipy.stats import wilcoxon

df = pd.read_csv("runs.csv")

hh = df[df["tags.approach"]=="hyperheuristic"]["metrics.best_score"]
opt = df[df["tags.approach"]=="optuna_baseline"]["metrics.best_score"]

stat, p = wilcoxon(hh, opt)
print(p)
