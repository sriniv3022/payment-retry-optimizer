import pandas as pd
import os


def load_data():

    df = pd.read_csv("data/raw/payments.csv")
    return df


def create_features(df):

    df["amount_bucket"] = pd.cut(
        df["amount"],
        bins=[0, 50, 100, 250, 500],
        labels=["low", "medium", "high", "very_high"]
    )

    df["is_high_retry_time"] = (df["hours_since_decline"] >= 24).astype(int)

    df = pd.get_dummies(df, columns=[
        "card_type",
        "issuer",
        "decline_reason",
        "amount_bucket"
    ])

    return df


def save_features(df):

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/features.csv", index=False)
    print("Features saved to data/processed/features.csv")


if __name__ == "__main__":

    df = load_data()
    df = create_features(df)
    save_features(df)