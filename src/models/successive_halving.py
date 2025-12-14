from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV

def successive_halving(model, param_dist, X, y, factor=3, scoring="f1_macro"):
    search = HalvingRandomSearchCV(
        estimator=model,
        param_distributions=param_dist,
        factor=factor,
        scoring=scoring,
        random_state=0
    )
    search.fit(X, y)
    return search.best_score_, search.best_params_
