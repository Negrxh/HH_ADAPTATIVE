from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

def preprocess_dataset(X, y, test_size=0.2, random_state=42):

    # 1. Split primero
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # 2. Variance threshold (fit SOLO en train)
    selector = VarianceThreshold(threshold=0)
    X_train = selector.fit_transform(X_train)
    X_test = selector.transform(X_test)

    # 3. Escalado (fit SOLO en train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
