import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def load_features():

    df = pd.read_csv("data/processed/features.csv")
    return df


def train():

    df = load_features()

    X = df.drop(columns=["success", "transaction_id"])
    y = df["success"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LGBMClassifier()

    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, preds)

    print("AUC:", auc)

    joblib.dump(model, "model_retry.pkl")

    print("Model saved as model_retry.pkl")


if __name__ == "__main__":
    train()