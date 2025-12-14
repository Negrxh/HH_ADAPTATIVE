import pandas as pd


def extract_meta_features(X, y):

    # Convertir a DataFrame si es numpy
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    meta_features = {
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "feature_correlation_mean": X.corr().abs().mean().mean(),
        "feature_std_mean": X.std().mean(),
        "feature_skew_mean": X.skew().mean(),
        # ejemplo de meta-feature simple
        "class_balance": (
            y.value_counts(normalize=True).max() if hasattr(y, "value_counts") else None
        ),
        "feature_skewness": X.skew().mean(),
        "feature_kurtosis": X.kurtosis().mean(),
        "noise_indicator": 1 - X.mean().corr(y),  # aproximación simple
    }

    return meta_features
