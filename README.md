# Payment Retry Optimizer

A machine learning system that predicts the optimal retry time for declined payment transactions.

## Problem

Payment providers like Stripe or PayPal often retry declined transactions using simple heuristics (e.g., retry daily for 5 days).

This project builds a data-driven approach to select the retry time with the highest probability of success.

## Features

- Synthetic payment dataset generator
- Feature engineering pipeline
- Gradient Boosting model for retry success prediction
- Retry decision engine
- Strategy evaluation vs baseline heuristic
- Interactive Streamlit dashboard

## Tech Stack

Python  
Pandas  
LightGBM  
Scikit-Learn  
Streamlit  

## Dashboard

Interactive Streamlit dashboard for predicting optimal retry timing.
