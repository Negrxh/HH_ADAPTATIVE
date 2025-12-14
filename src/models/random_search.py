from sklearn.model_selection import RandomizedSearchCV

def random_search(model, param_dist, X, y, iters=15, cv=3, scoring="f1_macro"):
    search = RandomizedSearchCV(model, param_dist, n_iter=iters, cv=cv, scoring=scoring)
    search.fit(X, y)
    return search.best_score_, search.best_params_
