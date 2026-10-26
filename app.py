"""
app.py — Streamlit Interactive Frontend for XAI Credit Risk Assessment.

• Collects applicant details via a premium‑styled sidebar form.
• Runs the trained XGBoost model to predict Approved / Rejected.
• Uses LIME to generate a local explanation and renders the visualization
  directly inside the Streamlit app.
"""

import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import lime
import lime.lime_tabular
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="XAI Credit Risk Assessment",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────────────────
# Custom CSS — premium glassmorphism dark theme
# ────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ───────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a40 40%, #302b63 100%);
}

/* ── Sidebar ──────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.85);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] .stMarkdown h2 {
    background: linear-gradient(90deg, #00d2ff, #7b2ff7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
}

/* ── Cards ────────────────────────────────────────────────────────────── */
.glass-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.glass-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(123, 47, 247, 0.25);
}

/* ── Result badges ────────────────────────────────────────────────────── */
.result-approved {
    background: linear-gradient(135deg, #00b09b, #96c93d);
    color: #fff;
    padding: 1.6rem 2rem;
    border-radius: 16px;
    text-align: center;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 1px;
    box-shadow: 0 8px 30px rgba(0, 176, 155, 0.35);
    animation: pulse-green 2s infinite;
}
.result-rejected {
    background: linear-gradient(135deg, #e53935, #ff6f00);
    color: #fff;
    padding: 1.6rem 2rem;
    border-radius: 16px;
    text-align: center;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 1px;
    box-shadow: 0 8px 30px rgba(229, 57, 53, 0.35);
    animation: pulse-red 2s infinite;
}
@keyframes pulse-green {
    0%, 100% { box-shadow: 0 8px 30px rgba(0,176,155,0.35); }
    50% { box-shadow: 0 8px 45px rgba(150,201,61,0.55); }
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 8px 30px rgba(229,57,53,0.35); }
    50% { box-shadow: 0 8px 45px rgba(255,111,0,0.55); }
}

/* ── Metric tiles ─────────────────────────────────────────────────────── */
.metric-tile {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    text-align: center;
}
.metric-tile .label { color: #aaa; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
.metric-tile .value { color: #fff; font-size: 1.8rem; font-weight: 700; }

/* ── Headers ──────────────────────────────────────────────────────────── */
.hero-title {
    text-align: center;
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00d2ff, #7b2ff7, #ff6f00);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero-sub {
    text-align: center;
    color: #8a8aad;
    font-size: 1rem;
    margin-bottom: 2rem;
}
.section-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #e0e0ff;
    margin-bottom: 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid rgba(123, 47, 247, 0.4);
}

/* ── Fix dark‑theme matplotlib background ─────────────────────────────── */
.stPlotlyChart, .stPyplot {
    background: transparent !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────────────────
# Load artifacts
# ────────────────────────────────────────────────────────────────────────────
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(ARTIFACT_DIR, "xgb_model.joblib"))
    scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.joblib"))
    le_gender = joblib.load(os.path.join(ARTIFACT_DIR, "le_gender.joblib"))
    le_defaults = joblib.load(os.path.join(ARTIFACT_DIR, "le_defaults.joblib"))
    oe_education = joblib.load(os.path.join(ARTIFACT_DIR, "oe_education.joblib"))
    feature_names = joblib.load(os.path.join(ARTIFACT_DIR, "feature_names.joblib"))
    return model, scaler, le_gender, le_defaults, oe_education, feature_names


@st.cache_data
def load_training_data():
    """Load training data for LIME — it needs a representative background set."""
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "loan_data.csv"))
    return df


try:
    model, scaler, le_gender, le_defaults, oe_education, feature_names = load_artifacts()
except FileNotFoundError:
    st.error("⚠️ Model artifacts not found. Run `python train_model.py` first.")
    st.stop()

# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────
EDUCATION_OPTIONS = ["High School", "Associate", "Bachelor", "Master", "Doctorate"]
HOME_OWNERSHIP_OPTIONS = ["RENT", "OWN", "MORTGAGE", "OTHER"]
LOAN_INTENT_OPTIONS = [
    "PERSONAL",
    "EDUCATION",
    "MEDICAL",
    "VENTURE",
    "HOMEIMPROVEMENT",
    "DEBTCONSOLIDATION",
]
NUMERIC_COLS = [
    "person_age",
    "person_income",
    "person_emp_exp",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
]

# ────────────────────────────────────────────────────────────────────────────
# Hero header
# ────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🏦 XAI Credit Risk Assessment</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Explainable AI‑Powered Loan Approval Prediction with LIME Explanations</p>',
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────────────────
# Sidebar — applicant form
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Applicant Details")
    st.markdown("---")

    person_age = st.slider("Age", 18, 80, 28, help="Applicant's age in years")
    person_gender = st.selectbox("Gender", ["female", "male"])
    person_education = st.selectbox("Education Level", EDUCATION_OPTIONS, index=2)
    person_income = st.number_input("Annual Income ($)", 8_000, 1_000_000, 55_000, step=1_000)
    person_emp_exp = st.slider("Employment Experience (years)", 0, 50, 3)
    person_home_ownership = st.selectbox("Home Ownership", HOME_OWNERSHIP_OPTIONS)

    st.markdown("---")
    st.markdown("## 💰 Loan Details")
    loan_amnt = st.number_input("Loan Amount ($)", 500, 100_000, 15_000, step=500)
    loan_intent = st.selectbox("Loan Intent", LOAN_INTENT_OPTIONS)
    loan_int_rate = st.slider("Interest Rate (%)", 5.0, 25.0, 11.0, step=0.01)
    loan_percent_income = st.slider("Loan % of Income", 0.0, 1.0, 0.20, step=0.01)

    st.markdown("---")
    st.markdown("## 📊 Credit History")
    cb_person_cred_hist_length = st.slider("Credit History Length (years)", 2, 30, 5)
    credit_score = st.slider("Credit Score", 300, 850, 650)
    previous_loan_defaults = st.selectbox("Previous Loan Defaults?", ["No", "Yes"])

    st.markdown("---")
    predict_btn = st.button("🚀 Predict & Explain", use_container_width=True, type="primary")


# ────────────────────────────────────────────────────────────────────────────
# Preprocessing helper
# ────────────────────────────────────────────────────────────────────────────
def preprocess_input(
    age, gender, education, income, emp_exp, home, loan_amt, intent,
    int_rate, pct_income, cred_hist, cred_score, defaults,
):
    """Mirror the exact preprocessing from train_model.py."""
    row = {}

    # Numeric features
    row["person_age"] = age
    row["person_income"] = income
    row["person_emp_exp"] = emp_exp
    row["loan_amnt"] = loan_amt
    row["loan_int_rate"] = int_rate
    row["loan_percent_income"] = pct_income
    row["cb_person_cred_hist_length"] = cred_hist
    row["credit_score"] = cred_score

    # Binary encode
    row["person_gender"] = le_gender.transform([gender])[0]
    row["previous_loan_defaults_on_file"] = le_defaults.transform([defaults])[0]

    # Ordinal encode education
    row["person_education"] = int(oe_education.transform([[education]])[0, 0])

    # One‑hot encode home ownership
    for cat in HOME_OWNERSHIP_OPTIONS:
        row[f"person_home_ownership_{cat}"] = 1 if home == cat else 0

    # One‑hot encode loan intent
    for cat in LOAN_INTENT_OPTIONS:
        row[f"loan_intent_{cat}"] = 1 if intent == cat else 0

    # Build dataframe with correct column order
    df_input = pd.DataFrame([row])[feature_names]

    # Scale numeric columns
    df_input[NUMERIC_COLS] = scaler.transform(df_input[NUMERIC_COLS])

    return df_input


def build_lime_training_set():
    """Prepare training data in the same encoded+scaled format for LIME."""
    df = load_training_data()
    df = df[df["person_age"] <= 100].reset_index(drop=True)

    X = df.drop(columns=["loan_status"])

    # Binary
    X["person_gender"] = le_gender.transform(X["person_gender"])
    X["previous_loan_defaults_on_file"] = le_defaults.transform(
        X["previous_loan_defaults_on_file"]
    )

    # Ordinal
    X["person_education"] = oe_education.transform(X[["person_education"]]).astype(int)

    # One‑hot
    X = pd.get_dummies(
        X,
        columns=["person_home_ownership", "loan_intent"],
        drop_first=False,
        dtype=int,
    )

    # Ensure correct column order
    X = X[feature_names]

    # Scale
    X[NUMERIC_COLS] = scaler.transform(X[NUMERIC_COLS])

    return X.values


# ────────────────────────────────────────────────────────────────────────────
# Prediction & explanation
# ────────────────────────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner("Running model inference…"):
        input_df = preprocess_input(
            person_age, person_gender, person_education, person_income,
            person_emp_exp, person_home_ownership, loan_amnt, loan_intent,
            loan_int_rate, loan_percent_income, cb_person_cred_hist_length,
            credit_score, previous_loan_defaults,
        )

        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        prob_approved = probabilities[0]
        prob_rejected = probabilities[1]

    # ── Result section ──────────────────────────────────────────────────
    col_result, col_space = st.columns([3, 1])
    with col_result:
        if prediction == 0:
            st.markdown(
                f'<div class="result-approved">✅ LOAN APPROVED &nbsp;—&nbsp; Confidence {prob_approved:.1%}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="result-rejected">❌ LOAN REJECTED &nbsp;—&nbsp; Confidence {prob_rejected:.1%}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")

    # ── Probability gauges ──────────────────────────────────────────────
    st.markdown('<p class="section-title">📊 Prediction Probabilities</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="metric-tile"><div class="label">Approved</div>'
            f'<div class="value" style="color:#96c93d">{prob_approved:.1%}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-tile"><div class="label">Rejected</div>'
            f'<div class="value" style="color:#e53935">{prob_rejected:.1%}</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        label = "Approved" if prediction == 0 else "Rejected"
        color = "#96c93d" if prediction == 0 else "#e53935"
        st.markdown(
            f'<div class="metric-tile"><div class="label">Decision</div>'
            f'<div class="value" style="color:{color}">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Applicant summary ───────────────────────────────────────────────
    st.markdown('<p class="section-title">👤 Applicant Summary</p>', unsafe_allow_html=True)
    summary_data = {
        "Feature": [
            "Age", "Gender", "Education", "Annual Income", "Employment Exp.",
            "Home Ownership", "Loan Amount", "Loan Intent", "Interest Rate",
            "Loan % Income", "Credit History", "Credit Score", "Previous Defaults",
        ],
        "Value": [
            f"{person_age} years", person_gender.title(), person_education,
            f"${person_income:,.0f}", f"{person_emp_exp} years",
            person_home_ownership, f"${loan_amnt:,.0f}", loan_intent,
            f"{loan_int_rate:.2f}%", f"{loan_percent_income:.0%}",
            f"{cb_person_cred_hist_length} years", credit_score,
            previous_loan_defaults,
        ],
    }
    st.dataframe(
        pd.DataFrame(summary_data),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("")

    # ── LIME explanation ────────────────────────────────────────────────
    st.markdown(
        '<p class="section-title">🔍 LIME Local Explanation</p>',
        unsafe_allow_html=True,
    )
    st.caption(
        "LIME (Local Interpretable Model‑agnostic Explanations) shows which features "
        "pushed the decision toward Approved or Rejected for **this specific applicant**."
    )

    with st.spinner("Generating LIME explanation — this may take a moment…"):
        # Build background training set for LIME
        training_data = build_lime_training_set()

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            class_names=["Approved", "Rejected"],
            mode="classification",
            random_state=42,
        )

        instance = input_df.values[0]

        explanation = explainer.explain_instance(
            instance,
            model.predict_proba,
            num_features=12,
            top_labels=2,
        )

    # ── Render LIME figure directly ─────────────────────────────────────
    predicted_label = int(prediction)
    fig = explanation.as_pyplot_figure(label=predicted_label)
    fig.set_size_inches(10, 6)
    fig.patch.set_facecolor("#1a1a40")
    fig.patch.set_alpha(0.9)

    ax = fig.gca()
    ax.set_facecolor("#1a1a40")
    ax.tick_params(colors="#e0e0ff")
    ax.xaxis.label.set_color("#e0e0ff")
    ax.yaxis.label.set_color("#e0e0ff")
    ax.title.set_color("#e0e0ff")
    for spine in ax.spines.values():
        spine.set_color("#444")

    class_label = "Rejected" if predicted_label == 1 else "Approved"
    ax.set_title(f"LIME Explanation — Prediction: {class_label}", fontsize=14, fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Feature importance table from LIME ──────────────────────────────
    st.markdown("")
    st.markdown(
        '<p class="section-title">📋 Feature Contributions</p>',
        unsafe_allow_html=True,
    )

    lime_list = explanation.as_list(label=predicted_label)
    lime_df = pd.DataFrame(lime_list, columns=["Feature Rule", "Contribution"])
    lime_df["Direction"] = lime_df["Contribution"].apply(
        lambda x: "🟢 Supports" if (x > 0 and predicted_label == 0) or (x < 0 and predicted_label == 1)
        else "🔴 Opposes"
    )
    lime_df["Contribution"] = lime_df["Contribution"].apply(lambda x: f"{x:+.4f}")

    st.dataframe(lime_df, use_container_width=True, hide_index=True)

    # ── Intercept / base value info ─────────────────────────────────────
    st.markdown("")
    st.info(
        f"ℹ️ **LIME intercept** for class *{class_label}*: "
        f"`{explanation.intercept[predicted_label]:.4f}` — this is the base prediction "
        f"before feature contributions are added."
    )

else:
    # ── Landing state ───────────────────────────────────────────────────
    st.markdown("")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(
            '<div class="glass-card"><h3>🤖 XGBoost Model</h3>'
            "<p style='color:#aaa'>Industry‑grade gradient boosting trained on 45,000+ "
            "loan applications with class‑balanced sampling.</p></div>",
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            '<div class="glass-card"><h3>🔍 LIME Explanations</h3>'
            "<p style='color:#aaa'>Every prediction comes with a transparent, "
            "human‑readable breakdown of the factors that drove the decision.</p></div>",
            unsafe_allow_html=True,
        )
    with col_c:
        st.markdown(
            '<div class="glass-card"><h3>⚡ Real‑Time Inference</h3>'
            "<p style='color:#aaa'>Adjust applicant details in the sidebar and get "
            "instant predictions with full explainability.</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown(
        '<div style="text-align:center;color:#666;padding:2rem">'
        "👈 Fill in applicant details in the sidebar and click <b>Predict & Explain</b>"
        "</div>",
        unsafe_allow_html=True,
    )
