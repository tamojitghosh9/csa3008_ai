# XAI Credit Risk Assessment Framework

Explainable Artificial Intelligence Framework for Credit Risk Assessment and Loan Approval Prediction.

## Quick Start

```bash
# 1. Create & activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python train_model.py

# 4. Launch the Streamlit app
streamlit run app.py
```

## Project Structure

```
├── loan_data.csv          # Raw dataset (45,000 loan applications)
├── requirements.txt       # Pinned Python dependencies
├── train_model.py         # XGBoost model training pipeline
├── app.py                 # Streamlit interactive frontend
├── artifacts/             # Saved model & preprocessors (auto-generated)
│   ├── xgb_model.joblib
│   ├── scaler.joblib
│   ├── le_gender.joblib
│   ├── le_defaults.joblib
│   ├── oe_education.joblib
│   └── feature_names.joblib
└── README.md
```

## Features

| Feature | Description |
|---------|-------------|
| **XGBoost Classifier** | Gradient-boosted trees with class-imbalance handling |
| **LIME Explanations** | Local interpretable explanations for each prediction |
| **Streamlit Dashboard** | Premium glassmorphism UI with real-time inference |
| **14 Input Features** | Age, income, credit score, loan intent, and more |

## Tech Stack

- **Python 3.12** + venv
- **XGBoost** — gradient boosting classifier
- **LIME** — local interpretable model-agnostic explanations
- **Streamlit** — interactive web frontend
- **scikit-learn** — preprocessing & evaluation
- **pandas / numpy / matplotlib** — data handling & visualization
