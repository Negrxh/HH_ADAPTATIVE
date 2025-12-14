import numpy as np


class HyperHeuristicController:
    def __init__(self, heuristics, meta_features, epsilon=0.2, seed=42):
        self.heuristics = heuristics
        self.meta = meta_features
        self.epsilon = epsilon


        # RNG LOCAL (clave para reproducibilidad)
        # SEED PARA PODER EJECUTAR VARIAS VECES
        self.rng = np.random.default_rng(seed)

        # Valor de cada heurística (score esperado)
        self.values = {h: 0.0 for h in heuristics}
        self.counts = {h: 0 for h in heuristics}

        # Sesgo inicial según meta-features (importante!)
        self.bias = self._compute_bias()

    def _compute_bias(self):
        bias = {}
        n_samples = self.meta.get("n_samples", 0)
        n_features = self.meta.get("n_features", 0)

        for action in self.heuristics:  # action es ej: "svm_optuna"
            model, method = action.split("_")
            score_bias = 0.0

            # === BIAS POR MODELO (Conocimiento Experto) ===

            # SVM sufre con muchos datos (O(n^3))
            if model == "svm":
                if n_samples > 2000:
                    score_bias -= 0.1
                else:
                    score_bias += 0.05  # Bueno para pocos datos

            # RF es robusto y generalista
            if model == "rf":
                score_bias += 0.02

            # KNN sufre con alta dimensionalidad (maldición de la dimensión)
            if model == "knn" and n_features > 50:
                score_bias -= 0.05

            # === BIAS POR HEURÍSTICA (Lo que ya tenías) ===
            if method == "optuna":
                score_bias -= (
                    0.001 * n_samples
                )  # Optuna lento en inicializar si el loop es muy pesado
            elif method == "random":
                score_bias += 0.001 * n_features

            bias[action] = score_bias

        return bias

    # Exploration con UCB
    def select(self):
        total_uses = sum(self.counts[h] for h in self.heuristics) + 1

        def ucb(h):
            if self.counts[h] == 0:
                return float("inf")
            exploit = self.values[h] + self.bias[h]
            explore = np.sqrt(np.log(total_uses) / self.counts[h])
            return exploit + 0.1 * explore  # α = 0.1 tunable

        return max(self.heuristics, key=ucb)

    def update(self, h, score):

        reward = (score - 0.5) * 2  # convertir a rango [-1, 1]

        self.counts[h] += 1
        n = self.counts[h]

        # incremental mean update
        self.values[h] += (reward - self.values[h]) / n

    def get_state(self):
        return {
            "values": dict(self.values),
            "counts": dict(self.counts),
            "bias": dict(self.bias),
        }
