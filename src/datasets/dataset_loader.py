import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_wine,
    load_iris,
    fetch_openml
)

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))  

# DATASET QUE SE TIENEN
AVAILABLE_DATASETS = {
    "breast_cancer": load_breast_cancer,
    "wine": load_wine,
    "iris": load_iris,
}

# name: interno en AVAILABLE_DATASETS
# path: externo con URL
def load_dataset(name, path=None):

    # INTERNO
    if name in AVAILABLE_DATASETS:
        data = AVAILABLE_DATASETS[name](as_frame=True)
        X, y = data.data, data.target
        return X, y
    

    if name == "mnist":
        X, y = fetch_openml(
            "mnist_784",
            version=1,
            return_X_y=True,
            as_frame=True
        )

        y = y.astype(int)

        subsample = 3000
        random_state = 42

        if subsample is not None:
            from sklearn.model_selection import train_test_split
            X, _, y, _ = train_test_split(
                X,
                y,
                train_size=subsample,
                stratify=y,
                random_state=random_state
            )

        return X, y
    

    # EXTERNO
    if path is not None:
        full_path = os.path.join(PROJECT_ROOT, path)
        df = pd.read_csv(full_path)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        return X, y

    raise ValueError(f"Dataset {name} no encontrado")
