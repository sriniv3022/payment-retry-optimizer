import pandas as pd
import joblib

model = joblib.load("model_retry.pkl")

df = pd.read_csv("data/processed/features.csv")

X = df.drop(columns=["success", "transaction_id"])
y = df["success"]


def baseline_strategy():

    retries = df[df["hours_since_decline"] == 24]
    return retries["success"].mean()


def ml_strategy():

    preds = model.predict_proba(X)[:, 1]
    df["pred_prob"] = preds

    selected = df[df["pred_prob"] > 0.3]

    return selected["success"].mean()


print("Baseline success rate:", baseline_strategy())
print("ML strategy success rate:", ml_strategy())