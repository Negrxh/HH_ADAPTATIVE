import mlflow
import optuna
import numpy as np
import os
import random
import sys

# Ajusta estos imports según tu estructura
sys.path.append("./src") 
try:
    from models.model_config import get_model_config
    from datasets.dataset_loader import load_dataset
    from datasets.preprocessors import preprocess_dataset
except ImportError:
    sys.path.append("..")
    from models.model_config import get_model_config
    from datasets.dataset_loader import load_dataset
    from datasets.preprocessors import preprocess_dataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

# Nombre del experimento aislado
NEW_EXPERIMENT_NAME = "optuna_baseline_isolated_v2" 

def run_optuna_baseline(dataset_name, dataset_path=None, seed=42, n_trials=160):
    
    set_seed(seed)
    
    # Nombre para identificar que ahora es mixto
    run_name = f"[BASELINE CASH] {dataset_name} (s={seed})"

    with mlflow.start_run(run_name=run_name):
        
        # Tags actualizados
        mlflow.set_tag("tipo", "baseline_cash") # CASH = Model Selection + HPO
        mlflow.log_param("seed", seed)
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("strategy", "Optuna (RF + SVM)") # Importante para distinguir
        mlflow.log_param("budget", n_trials)

        print(f"--> Cargando {dataset_name}...")
        X, y = load_dataset(dataset_name, dataset_path)
        X_train, X_test, y_train, y_test = preprocess_dataset(X, y)

        # Callback para gráficas
        def mlflow_callback(study, trial):
            if trial.value is not None:
                step = trial.number
                mlflow.log_metric("best_so_far", study.best_value, step=step)
                mlflow.log_metric("score", trial.value, step=step)

        # --- FUNCIÓN OBJETIVO MODIFICADA (EL CEREBRO DE OPTUNA) ---
        def objective(trial):
            # 1. Optuna decide qué algoritmo usar en este intento
            classifier_name = trial.suggest_categorical("classifier", ["rf", "svm"])
            
            # 2. Obtenemos la configuración según el elegido
            # Asumo que tu get_model_config soporta ("svm", "optuna", trial)
            _, params = get_model_config(classifier_name, "optuna", trial=trial)
            
            # 3. Instanciamos el modelo correcto
            if classifier_name == "rf":
                params["random_state"] = seed
                model = RandomForestClassifier(**params)
            else: # SVM
                # SVM suele necesitar random_state si probability=True o linear, 
                # pero SVC por defecto en sklearn lo acepta.
                if "random_state" in SVC().get_params():
                    params["random_state"] = seed
                model = SVC(**params)

            # 4. Evaluación
            score = np.mean(
                cross_val_score(model, X_train, y_train, cv=3, scoring="f1_macro")
            )
            
            # Guardamos qué modelo eligió Optuna en este paso (útil para análisis)
            trial.set_user_attr("model_selected", classifier_name)
            
            return score

        # Ejecución
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        
        study.optimize(objective, n_trials=n_trials, callbacks=[mlflow_callback])

        # Métricas finales
        mlflow.log_metric("final_f1_macro", study.best_value)
        mlflow.log_params(study.best_params)
        
        # Guardamos qué modelo ganó al final
        mlflow.log_param("winning_model", study.best_params["classifier"])

        print(f"✅ Fin: {dataset_name} (s={seed}) | Best: {study.best_value:.4f} | Ganó: {study.best_params['classifier']}")


if __name__ == "__main__":
    mlflow.set_tracking_uri("file:./mlruns")
    
    print(f"Configurando experimento: {NEW_EXPERIMENT_NAME}")
    mlflow.set_experiment(NEW_EXPERIMENT_NAME)
    
    DATASETS = [
        # {"name": "mnist", "path": None},
        # {"name": "iris", "path": None}, 
        # {"name": "wine", "path": None}, 
        {"name": "breast_cancer", "path": None}, 
    ]
    
    SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    PRESUPUESTO = 20 

    print("🚀 Iniciando Baseline CASH (RF + SVM)...")
    
    for conf in DATASETS:
        for seed in SEEDS:
            try:
                run_optuna_baseline(conf["name"], conf["path"], seed, PRESUPUESTO)
            except Exception as e:
                print(f"❌ Error en {conf['name']} seed {seed}: {e}")