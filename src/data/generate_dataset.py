import pandas as pd
import numpy as np
import os

np.random.seed(42)

N = 50000

def simulate_success(row):

    if row["decline_reason"] == "insufficient_funds":
        return np.random.binomial(1, 0.35 if row["hours_since_decline"] >= 24 else 0.1)

    elif row["decline_reason"] == "network_error":
        return np.random.binomial(1, 0.6 if row["hours_since_decline"] <= 6 else 0.25)

    elif row["decline_reason"] == "risk_review":
        return np.random.binomial(1, 0.15)

    return 0


def generate_data():

    data = pd.DataFrame({
        "transaction_id": np.arange(N),
        "amount": np.random.randint(10, 500, N),
        "card_type": np.random.choice(["credit", "debit"], N),
        "issuer": np.random.choice(["Chase", "Citi", "BoA", "Wells"], N),
        "decline_reason": np.random.choice(
            ["insufficient_funds", "network_error", "risk_review"], N),
        "hours_since_decline": np.random.choice([1, 6, 12, 24, 48], N)
    })

    data["success"] = data.apply(simulate_success, axis=1)

    os.makedirs("data/raw", exist_ok=True)
    data.to_csv("data/raw/payments.csv", index=False)

    print("Dataset generated at data/raw/payments.csv")


if __name__ == "__main__":
    generate_data()