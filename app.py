"""
Streamlit UI for the Bayesian Network heart disease prediction project.

Loads the expert-specified Bayesian Network, fits it to the discretized
Cleveland Heart Disease dataset, and allows interactive probabilistic
queries via Variable Elimination.

Usage:
    streamlit run app.py
"""

import os
import sys
import time

import streamlit as st
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — ensure project root is importable
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from preprocessing import load_and_prepare          # noqa: E402
from structure_learning import expert_structure, fit_parameters  # noqa: E402
from pgmpy.inference import VariableElimination      # noqa: E402

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease BN Predictor",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS — black / navy / steel-blue academic palette
# ---------------------------------------------------------------------------
_CSS = """
<style>
    /* ---- Main header ---- */
    .main h1 {
        color: #4682B4;
        font-weight: 700;
        border-bottom: 2px solid #1B2A4A;
        padding-bottom: 0.4rem;
    }
    .main h2, .main h3 {
        color: #5A8DB5;
    }

    /* ---- Metric cards ---- */
    [data-testid="stMetricValue"] {
        font-size: 2.6rem;
        font-weight: 700;
        color: #4682B4;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem;
        color: #8899AA;
    }

    /* ---- Sidebar header ---- */
    [data-testid="stSidebar"] h2 {
        color: #4682B4;
    }

    /* ---- Primary button ---- */
    .stButton > button[kind="primary"],
    .stButton > button {
        background-color: #1B2A4A;
        border: 1px solid #4682B4;
        color: white;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #4682B4;
        border-color: #4682B4;
        color: white;
    }

    /* ---- Dividers ---- */
    hr { border-color: #1B2A4A; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Discretized variable categories — mirrors preprocessing.py exactly
# ---------------------------------------------------------------------------
EVIDENCE_CATEGORIES = {
    "Age":               ["<40", "40-49", "50-59", "60+"],
    "Sex":               ["Male", "Female"],
    "ChestPainType":     ["TypicalAngina", "AtypicalAngina",
                          "NonAnginal", "Asymptomatic"],
    "RestingBP":         ["Normal(<120)", "Elevated(120-139)", "High(140+)"],
    "Cholesterol":       ["Desirable(<200)", "Borderline(200-239)",
                          "High(240+)"],
    "FastingBloodSugar": [">120mg/dl", "<=120mg/dl"],
    "RestingECG":        ["Normal", "ST-T_Abnormality", "LV_Hypertrophy"],
    "MaxHeartRate":      ["Low(<120)", "Moderate(120-149)", "High(150+)"],
    "ExerciseAngina":    ["Yes", "No"],
    "STDepression":      ["None(0)", "Mild(0-1)", "Moderate(1-2)",
                          "Severe(2+)"],
    "STSlope":           ["Upsloping", "Flat", "Downsloping"],
    "MajorVessels":      ["0", "1", "2", "3"],
    "Thalassemia":       ["Normal", "Fixed_or_Other", "Reversible"],
}

_UNSELECTED = "-- Unknown --"

# ---------------------------------------------------------------------------
# Model loading — cached so it only runs once per session
# ---------------------------------------------------------------------------
@st.cache_resource
def _build_model():
    """Load CSV, discretize, build expert BN, fit parameters."""
    data_path = os.path.join(_ROOT, "data", "heart.csv")
    _, data = load_and_prepare(data_path)
    model = expert_structure()
    model = fit_parameters(model, data)
    return model


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
def main():
    st.title("Heart Disease Risk Prediction")
    st.caption(
        "Expert-specified Bayesian Network over the UCI Cleveland dataset "
        "with exact inference via Variable Elimination (pgmpy)."
    )

    model = _build_model()
    engine = VariableElimination(model)

    # ---- Sidebar: patient evidence inputs --------------------------------
    st.sidebar.header("Patient Evidence")
    st.sidebar.markdown(
        "Set each attribute to the observed value, or leave as "
        "*Unknown* to treat it as unobserved."
    )

    evidence: dict[str, str] = {}
    for var_name, categories in EVIDENCE_CATEGORIES.items():
        options = [_UNSELECTED] + categories
        choice = st.sidebar.selectbox(var_name, options, index=0)
        if choice != _UNSELECTED:
            evidence[var_name] = choice

    # ---- Predict button --------------------------------------------------
    predict = st.sidebar.button("Predict", type="primary")

    if predict:
        # Run Variable Elimination query
        t0 = time.perf_counter()
        result = engine.query(
            variables=["HeartDisease"],
            evidence=evidence,
            show_progress=False,
        )
        query_ms = (time.perf_counter() - t0) * 1000.0

        # Extract posterior probabilities
        states = result.state_names["HeartDisease"]
        values = result.values
        prob = {s: float(v) for s, v in zip(states, values)}

        p_present = prob.get("Present", 0.0)
        p_absent  = prob.get("Absent",  0.0)

        # ---- Display results ---------------------------------------------
        st.markdown("---")
        st.subheader("P(HeartDisease | Evidence)")

        col_present, col_absent = st.columns(2)
        with col_present:
            st.metric("Present", f"{p_present * 100:.1f} %")
        with col_absent:
            st.metric("Absent", f"{p_absent * 100:.1f} %")

        # Probability bar chart
        chart_df = pd.DataFrame(
            {"Probability": [p_present, p_absent]},
            index=["Present", "Absent"],
        )
        st.bar_chart(chart_df)

        # Evidence summary
        if evidence:
            st.markdown("**Observed evidence:**")
            st.code(
                ", ".join(f"{k} = {v}" for k, v in evidence.items()),
                language="",
            )
        else:
            st.markdown("*No evidence provided — showing prior distribution.*")

        # ---- Inference metadata ------------------------------------------
        st.markdown("---")
        st.caption(
            f"Inference method: Variable Elimination  |  "
            f"Query time: {query_ms:.2f} ms"
        )

    else:
        st.info(
            "Configure patient attributes in the sidebar and press "
            "**Predict** to compute P(HeartDisease | evidence)."
        )

    # ---- Disclaimer ------------------------------------------------------
    st.markdown("---")
    st.caption(
        "Disclaimer: This is a course project demonstration "
        "(AAU, MSc AI, Probabilistic Graphical Models) "
        "and is not intended as a medical diagnostic tool."
    )


if __name__ == "__main__":
    main()
