import numpy as np
from sklearn.model_selection import cross_val_score

def hyperband_simple(model_class, param_sampler, X, y, R=27, η=3):
    """
    R = recurso máximo (ej: número de estimadores)
    param_sampler = lambda: dict con hiperparámetros
    """
    best_score = -1
    best_params = None

    # numero de brackets
    s_max = int(np.log(R) / np.log(η))

    for s in range(s_max + 1):
        n = int(np.ceil((s_max + 1) * η**s / (s + 1)))
        r = R * η**(-s)

        # generar n configuraciones
        configs = [param_sampler() for _ in range(n)]
        scores = []

        for i in range(s + 1):
            n_i = n * η**(-i)
            r_i = int(r * η**i)

            results = []
            for cfg in configs:
                model = model_class(**cfg)
                sc = cross_val_score(model, X, y, cv=3, scoring="f1_macro").mean()
                results.append((sc, cfg))

            # quedarnos con top-k
            results.sort(reverse=True, key=lambda x: x[0])
            configs = [cfg for _, cfg in results[:max(1, int(n_i / η))]]

        top_score, top_params = results[0]
        if top_score > best_score:
            best_score, best_params = top_score, top_params

    return best_score, best_params
