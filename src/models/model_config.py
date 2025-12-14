from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def get_model_config(model_name, method_name, trial=None):
    """
    Retorna (modelo_instanciado, espacio_de_busqueda)
    Adaptado para que funcione tanto con Sklearn (dict) como con Optuna (trial).
    """
    
    # === RANDOM FOREST ===
    if model_name == "rf":
        model = RandomForestClassifier()
        if method_name in ["random", "sh"]:
            params = {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 20, None],
                "criterion": ["gini", "entropy"]
            }
        elif method_name in ["optuna", "hb"]:
            # Para métodos que requieren sampling dinámico
            # Si es Optuna pasamos el 'trial', si es HB usamos numpy random
            if method_name == "optuna" and trial:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                    "max_depth": trial.suggest_int("max_depth", 5, 30),
                    "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"])
                }
            else: # HB o fallback
                params = {
                    "n_estimators": np.random.randint(50, 200),
                    "max_depth": np.random.randint(5, 30),
                    "criterion": np.random.choice(["gini", "entropy"])
                }
        return model, params

    # === SVM (Ideal para pocos datos) ===
    elif model_name == "svm":
        model = SVC(probability=True) # probability=True a veces necesario para log_loss pero lento
        if method_name in ["random", "sh"]:
            params = {
                "C": [0.1, 1, 10, 100],
                "kernel": ["linear", "rbf"]
            }
        elif method_name in ["optuna", "hb"]:
            if method_name == "optuna" and trial:
                params = {
                    "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
                    "kernel": trial.suggest_categorical("kernel", ["linear", "rbf"])
                }
            else:
                params = {
                    "C": 10 ** np.random.uniform(-3, 2),
                    "kernel": np.random.choice(["linear", "rbf"])
                }
        return model, params
    
    return None, None