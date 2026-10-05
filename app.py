import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent

st.set_page_config(page_title="Salary Predictor", page_icon="💰", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(BASE / "model" / "model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv(BASE / "data" / "Salary_Data.csv")


@st.cache_data
def load_metrics():
    with open(BASE / "model" / "metrics.json") as f:
        return json.load(f)


model = load_model()
df = load_data()
metrics = load_metrics()

st.title("💰 Salary Prediction App")
st.write(
    "Enter your years of experience and this app predicts an estimated salary "
    "using a **Linear Regression** model trained on a salary dataset."
)

tab_predict, tab_data, tab_model = st.tabs(["Predict", "Dataset", "Model performance"])

# ---------------- Predict ----------------
with tab_predict:
    min_exp, max_exp = float(df["Experience Years"].min()), float(df["Experience Years"].max())
    years = st.number_input(
        "Years of experience",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=0.5,
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    if st.button("Predict salary", type="primary"):
        pred = float(model.predict(pd.DataFrame({"Experience Years": [years]}))[0])
        st.session_state.history.append({"Years of experience": years, "Predicted salary": round(pred, 2)})
        st.success(f"Estimated salary: **${pred:,.2f}**")
        if years < min_exp or years > max_exp:
            st.warning(
                f"The training data only covers {min_exp}–{max_exp} years of experience, "
                "so this prediction is an extrapolation and may be less reliable."
            )

        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.scatter(df["Experience Years"], df["Salary"], label="Data", alpha=0.7)
        line_x = pd.DataFrame({"Experience Years": [0.0, max(max_exp, years)]})
        ax.plot(line_x["Experience Years"], model.predict(line_x), color="red", label="Model")
        ax.scatter([years], [pred], color="green", s=120, zorder=5, label="Your prediction")
        ax.set_xlabel("Years of experience")
        ax.set_ylabel("Salary")
        ax.legend()
        st.pyplot(fig)

    if st.session_state.history:
        st.subheader("Prediction history")
        st.dataframe(pd.DataFrame(st.session_state.history))
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

# ---------------- Dataset ----------------
with tab_data:
    st.subheader("Dataset information")
    st.write(f"Rows: **{df.shape[0]}**, Columns: **{df.shape[1]}**")
    st.dataframe(df)
    st.write("Summary statistics")
    st.dataframe(df.describe())
    st.write("Salary vs experience")
    st.scatter_chart(df, x="Experience Years", y="Salary")

# ---------------- Model ----------------
with tab_model:
    st.subheader("Model performance (test set)")
    c1, c2 = st.columns(2)
    c1.metric("R² score", f"{metrics['R2']:.3f}")
    c2.metric("MAE", f"${metrics['MAE']:,.0f}")
    c3, c4 = st.columns(2)
    c3.metric("RMSE", f"${metrics['RMSE']:,.0f}")
    c4.metric("MSE", f"{metrics['MSE']:,.0f}")
    st.caption(
        f"Each extra year of experience adds about ${model.coef_[0]:,.0f} to the predicted salary "
        f"(intercept: ${model.intercept_:,.0f})."
    )
