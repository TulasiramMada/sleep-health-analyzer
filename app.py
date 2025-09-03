import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sleep Health Analyzer", page_icon="😴", layout="wide")

st.title("😴 AI-Powered Sleep Health Analyzer")
st.write("Analyze lifestyle & sleep patterns, classify sleep quality, and get personalized tips.")

@st.cache_data
def load_data():
    return pd.read_csv("sample_data.csv")

@st.cache_resource
def train_model(df):
    features = ["age","sleep_hours","caffeine_mg","screen_time_hours","exercise_minutes",
                "stress_level","sleep_latency_minutes","awakenings"]
    X = df[features]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return clf, acc, features

tab1, tab2, tab3 = st.tabs(["Personal Assessment", "Dataset Analysis", "Model & EDA"])

with tab1:
    st.subheader("Personal Assessment")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=16, max_value=90, value=28, step=1)
        sleep_hours = st.number_input("Average Sleep Hours", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
        awakenings = st.number_input("Awakenings per Night", min_value=0, max_value=12, value=1, step=1)
    with col2:
        caffeine_mg = st.number_input("Daily Caffeine (mg)", min_value=0, max_value=1000, value=150, step=10)
        screen_time_hours = st.number_input("Evening Screen Time (hrs)", min_value=0.0, max_value=12.0, value=3.0, step=0.1)
        sleep_latency_minutes = st.number_input("Sleep Latency (min)", min_value=0, max_value=180, value=20, step=5)
    with col3:
        exercise_minutes = st.number_input("Daily Exercise (min)", min_value=0, max_value=300, value=20, step=5)
        stress_level = st.slider("Stress Level (1=Low, 5=High)", min_value=1, max_value=5, value=3)

    df = load_data()
    clf, acc, feat = train_model(df)

    if st.button("Analyze My Sleep"):
        X_user = pd.DataFrame([{
            "age": age,
            "sleep_hours": sleep_hours,
            "caffeine_mg": caffeine_mg,
            "screen_time_hours": screen_time_hours,
            "exercise_minutes": exercise_minutes,
            "stress_level": stress_level,
            "sleep_latency_minutes": sleep_latency_minutes,
            "awakenings": awakenings,
        }])

        pred = clf.predict(X_user)[0]
        proba = clf.predict_proba(X_user)[0]
        labels = clf.classes_
        st.markdown(f"### Predicted Sleep Quality: **{pred}**")
        st.caption(f"Model reference accuracy on sample data: {acc:.2%}")

        fig, ax = plt.subplots()
        ax.bar(labels, proba)
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Confidence")
        st.pyplot(fig)

        tips = []
        if sleep_hours < 7: tips.append("Aim for 7–9 hours of sleep by setting a consistent bedtime.")
        if caffeine_mg > 200: tips.append("Reduce caffeine intake after 2 PM to lower sleep latency.")
        if screen_time_hours > 2: tips.append("Limit screens 1 hour before bed or use blue-light filters.")
        if exercise_minutes < 20: tips.append("Light daytime exercise can improve sleep depth.")
        if stress_level >= 4: tips.append("Try relaxation techniques (breathing, journaling) before bed.")
        if sleep_latency_minutes > 30: tips.append("Consider a wind-down routine and reduce late caffeine.")
        if awakenings >= 3: tips.append("Review evening fluids and room temperature; consult a clinician if persistent.")
        st.markdown("#### Personalized Suggestions")
        if tips:
            for t in tips:
                st.write(f"- {t}")
        else:
            st.write("- Your inputs look great. Keep your routine consistent.")

with tab2:
    st.subheader("Dataset Analysis & Batch Scoring")
    uploaded = st.file_uploader("Upload CSV (optional). If empty, sample_data.csv will be used.", type=["csv"])
    if uploaded:
        data = pd.read_csv(uploaded)
    else:
        data = load_data()

    st.write("Preview:", data.head())

    required_cols = ["age","sleep_hours","caffeine_mg","screen_time_hours","exercise_minutes",
                     "stress_level","sleep_latency_minutes","awakenings"]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        clf, acc, features = train_model(load_data())
        preds = clf.predict(data[features])
        data_out = data.copy()
        data_out["prediction"] = preds
        st.write("Scored Sample:", data_out.head())
        st.download_button("⬇️ Download Scored CSV", data_out.to_csv(index=False).encode("utf-8"), file_name="scored_sleep_data.csv")

with tab3:
    st.subheader("Model Details & Exploratory Analysis")
    data = load_data()
    st.write("Sample Data Shape:", data.shape)

    st.write("Label Distribution:")
    st.write(data["label"].value_counts())

    cols_to_plot = ["sleep_hours","caffeine_mg","screen_time_hours","exercise_minutes","stress_level","sleep_latency_minutes","awakenings"]
    for c in cols_to_plot:
        fig, ax = plt.subplots()
        ax.hist(data[c], bins=20)
        ax.set_title(f"Distribution: {c}")
        st.pyplot(fig)

    st.caption("Note: Model trained on synthetic sample_data.csv for demo purposes. Replace with your dataset for production use.")
