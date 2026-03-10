import joblib
import pandas as pd

model = joblib.load("model_retry.pkl")

CANDIDATE_TIMES = [1, 6, 12, 24, 48]


def find_best_retry(transaction_features):

    best_time = None
    best_prob = 0

    for t in CANDIDATE_TIMES:

        features = transaction_features.copy()
        features["hours_since_decline"] = t

        df = pd.DataFrame([features])

        prob = model.predict_proba(df)[0][1]

        if prob > best_prob:
            best_prob = prob
            best_time = t

    return best_time, best_prob