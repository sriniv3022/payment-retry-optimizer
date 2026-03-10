import streamlit as st
import joblib
import pandas as pd

model = joblib.load("model_retry.pkl")

CANDIDATE_TIMES = [1, 6, 12, 24, 48]


def find_best_retry(features):

    best_time = None
    best_prob = 0

    for t in CANDIDATE_TIMES:

        temp = features.copy()
        temp["hours_since_decline"] = t

        df = pd.DataFrame([temp])

        prob = model.predict_proba(df)[0][1]

        if prob > best_prob:
            best_prob = prob
            best_time = t

    return best_time, best_prob


st.title("Smart Payment Retry Optimizer")

st.write("Predict optimal retry time for declined payments.")

amount = st.slider("Transaction Amount", 10, 500, 100)

card_type = st.selectbox(
    "Card Type",
    ["credit", "debit"]
)

issuer = st.selectbox(
    "Issuer Bank",
    ["Chase", "Citi", "BoA", "Wells"]
)

decline_reason = st.selectbox(
    "Decline Reason",
    ["insufficient_funds", "network_error", "risk_review"]
)

features = {
    "amount": amount,
    "hours_since_decline": 1,
    "is_high_retry_time": 0,

    "card_type_credit": int(card_type == "credit"),
    "card_type_debit": int(card_type == "debit"),

    "issuer_BoA": int(issuer == "BoA"),
    "issuer_Chase": int(issuer == "Chase"),
    "issuer_Citi": int(issuer == "Citi"),
    "issuer_Wells": int(issuer == "Wells"),

    "decline_reason_insufficient_funds": int(decline_reason == "insufficient_funds"),
    "decline_reason_network_error": int(decline_reason == "network_error"),
    "decline_reason_risk_review": int(decline_reason == "risk_review"),

    "amount_bucket_low": int(amount < 50),
    "amount_bucket_medium": int(50 <= amount < 100),
    "amount_bucket_high": int(100 <= amount < 250),
    "amount_bucket_very_high": int(amount >= 250)
}


if st.button("Predict Optimal Retry Time"):

    best_time, prob = find_best_retry(features)

    st.success(f"Optimal Retry Time: {best_time} hours")

    st.metric(
        "Predicted Success Probability",
        f"{prob:.2%}"
    )