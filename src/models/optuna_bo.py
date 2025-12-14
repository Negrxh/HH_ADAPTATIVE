import optuna
from sklearn.model_selection import cross_val_score
import numpy as np


def optuna_bo(model_class, sampler_fn, X, y, trials=20, seed=42):
    def objective(trial):
        params = sampler_fn(trial)
        model = model_class(**params)
        scores = cross_val_score(model, X, y, cv=3, scoring="f1_macro")
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=trials)
    return study.best_value, study.best_params
