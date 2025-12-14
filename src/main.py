from sklearn.datasets import load_breast_cancer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from controller import HyperHeuristicController
from models.random_search import random_search
from models.optuna_bo import optuna_bo
from models.successive_halving import successive_halving
from models.hyperband_simple import hyperband_simple
from models.model_config import get_model_config

from meta_features import extract_meta_features
from datasets.dataset_loader import load_dataset
from datasets.preprocessors import preprocess_dataset

import logging
import os
import csv
import time
import uuid

import optuna
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# PARA SEED
import random
import numpy as np


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


# SISTEMA DE LOGS CON MLFLOW
import mlflow

mlflow.set_experiment("hyperheuristic_experiment")


# SISTEMA DE LOGS
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/hh.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

available_models = ["rf", "svm"]
available_heuristics = ["random", "optuna", "sh", "hb"]


def run_experiment(dataset_name, dataset_path=None, seed=42):

    set_seed(seed)

    # MLF LOG
    mlflow.start_run()

    # LOGS INICIALES
    mlflow.log_param("seed", seed)
    run_id = str(uuid.uuid4())[:8]
    mlflow.log_param("run_id", run_id)
    mlflow.log_param("dataset", dataset_name)
    mlflow.log_param("n_rounds", 20)

    # === cargar dataset temporal ===
    X, y = load_dataset(dataset_name, dataset_path)
    X_train, X_test, y_train, y_test = preprocess_dataset(X, y)

    # EXTRACCION DE META-FEATURES
    meta = extract_meta_features(X_train, y_train)

    # MLF LOG
    mlflow.log_params(meta)

    # NUEVO DE HPO A CASH, AQUI SON LAS COMBINACIONES POSIBLES
    actions = [f"{m}_{h}" for m in available_models for h in available_heuristics]

    # CREACION DEL CONTROLADOR (HH)
    controller = HyperHeuristicController(
        heuristics=actions, meta_features=meta, seed=seed
    )

    best_score = -1
    best_params = None
    best_model_name = None

    # SISTEMA DE LOGS EN CSV
    csv_path = "logs/results.csv"
    csv_exists = os.path.isfile(csv_path)

    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)

    # si el archivo es nuevo => agregar cabecera
    if not csv_exists:
        csv_writer.writerow(
            [
                "run_id",
                "dataset",
                "round",
                "model",
                "heuristic",
                "score",
                "best_so_far",
                "round_time",
                "controller_values",
                "controller_counts",
                "controller_bias",
            ]
        )

    # PARA LOGS
    start_total = time.time()
    best_so_far = -1

    # BUCLE DE RONDAS
    for i in range(20):  # <--- Nº de rondas ajustable
        round_start = time.time()

        selected_action = controller.select()
        model_name, method_name = selected_action.split("_")

        logging.info(f"Ronda {i} — Modelo: {model_name} | Método: {method_name}")

        if method_name == "random":
            model, param_dist = get_model_config(model_name, "random")
            score, params = random_search(model, param_dist, X_train, y_train, iters=8)

        elif method_name == "optuna":
            # Para Optuna necesitamos una lambda que envuelva el get_model_config
            def sampler(t):
                _, p = get_model_config(model_name, "optuna", trial=t)
                return p

            # Necesitamos pasar la CLASE del modelo, no la instancia, a tu función optuna_bo
            model_class = get_model_config(model_name, "random")[0].__class__
            score, params = optuna_bo(model_class, sampler, X_train, y_train, trials=8, seed=seed)

        elif method_name == "sh":
            model, param_dist = get_model_config(model_name, "sh")
            score, params = successive_halving(model, param_dist, X_train, y_train)

        elif method_name == "hb":
            # HB en tu código espera una lambda para param_sampler
            model_inst, _ = get_model_config(model_name, "hb")
            model_class = model_inst.__class__

            param_sampler = lambda: get_model_config(model_name, "hb")[1]

            score, params = hyperband_simple(
                model_class, param_sampler, X_train, y_train
            )

        logging.info(f"Score obtenido: {score}")
        logging.info(f"Params: {params}")

        round_time = time.time() - round_start
        best_so_far = max(best_so_far, score)

        # MLF LOG
        mlflow.log_metric("score", score, step=i)
        mlflow.log_metric("best_so_far", best_so_far, step=i)
        mlflow.log_metric("round_time", round_time, step=i)
        mlflow.log_metric(f"used_{method_name}", 1, step=i)

        controller.update(selected_action, score)
        logging.info(f"Valores actualizados del controlador: {controller.values}")
        logging.info(f"Usos: {controller.counts}")

        state = controller.get_state()

        csv_writer.writerow(
            [
                run_id,
                dataset_name,
                i,
                model_name,
                method_name,
                score,
                best_so_far,
                round_time,
                state["values"],
                state["counts"],
                state["bias"],
            ]
        )

        if score > best_score:
            best_score, best_params, best_model_name = score, params, model_name

    logging.info(f"Ganador absoluto: {best_model_name} con params {best_params}")

    # 1. Obtenemos una instancia vacía del modelo ganador
    # Usamos 'random' como método dummy, solo queremos la instancia del modelo
    final_model, _ = get_model_config(best_model_name, "random")

    # 2. Le inyectamos los mejores parámetros encontrados
    final_model.set_params(**best_params)

    # 3. Entrenamos
    if "random_state" in final_model.get_params():
        final_model.set_params(random_state=seed)

    final_model.fit(X_train, y_train)
    preds = final_model.predict(X_test)

    # MLF LOG
    mlflow.log_metric("best_score", best_score)
    mlflow.log_params(best_params)
    mlflow.end_run()

    logging.info(f"BEST SCORE FINAL: {best_score}")
    logging.info(f"F1 FINAL TEST: {f1_score(y_test, preds, average='macro')}")

    csv_file.close()


def run_optuna_baseline(dataset_name, dataset_path=None, seed=42):

    set_seed(seed)

    mlflow.start_run(run_name="optuna_baseline")
    mlflow.log_param("seed", seed)

    X, y = load_dataset(dataset_name, dataset_path)
    X_train, X_test, y_train, y_test = preprocess_dataset(X, y)

    def objective(trial):
        _, params = get_model_config("rf", "optuna", trial=trial)
        params["random_state"] = seed

        model = RandomForestClassifier(**params)
        score = np.mean(
            cross_val_score(model, X_train, y_train, cv=3, scoring="f1_macro")
        )
        return score

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )

    study.optimize(objective, n_trials=160)

    mlflow.log_metric("best_score", study.best_value)
    mlflow.log_params(study.best_params)
    mlflow.log_param("method", "optuna_only")
    mlflow.log_param("budget", 20)

    mlflow.end_run()


if __name__ == "__main__":
    # run_experiment("ufc", dataset_path="data/fights_processed.csv")
    # run_experiment("iris")
    # run_optuna_baseline("iris")

    SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # SEEDS = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    # SEEDS = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    # SEEDS = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    for seed in SEEDS:
        run_experiment("mnist", seed=seed)
        run_optuna_baseline("mnist", seed=seed)

