"""Model training and validation utilities:

- LogisticRegression classifier
- Accuracy evaluation
- Coefficient inspection

"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
from pathlib import Path


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    y_predct = model.predict(X_test)
    acc_score = accuracy_score(y_pred=y_predct, y_true=y_test)
    return acc_score


def get_model_coffecients(model: LogisticRegression, feature_names=None) -> pd.DataFrame:
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "Coefficient": model.coef_[0]

    }).sort_values(by="Coefficient", ascending=False)
    return coef_df


def save_model(model, path: str) -> None:
    Path(path).parent.mkdir(exist_ok=True)
    joblib.dump(model, path)
