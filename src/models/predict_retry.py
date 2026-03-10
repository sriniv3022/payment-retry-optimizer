import pandas as pd
from src.decision_engine.retry_scheduler import find_best_retry


sample_transaction = {
    "amount": 120,
    "hours_since_decline": 1,
    "is_high_retry_time": 0,
    "card_type_credit": 1,
    "card_type_debit": 0,
    "issuer_BoA": 0,
    "issuer_Chase": 1,
    "issuer_Citi": 0,
    "issuer_Wells": 0,
    "decline_reason_insufficient_funds": 1,
    "decline_reason_network_error": 0,
    "decline_reason_risk_review": 0,
    "amount_bucket_low": 0,
    "amount_bucket_medium": 0,
    "amount_bucket_high": 1,
    "amount_bucket_very_high": 0
}


best_time, prob = find_best_retry(sample_transaction)

print("Best retry time:", best_time, "hours")
print("Predicted success probability:", prob)