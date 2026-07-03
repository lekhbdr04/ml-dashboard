import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score

st.set_page_config(page_title="ML Dashboard", layout="wide")
st.sidebar.title("📊 ML Dashboard")
page = st.sidebar.radio("Navigate", ["Home", "Dataset", "EDA", "Model Training", "Predictions"])

if page == "Home":
    st.title("🎯 Machine Learning Dashboard")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Regression Model")
        st.write("Predicts loan amount")
    with col2:
        st.subheader("✅ Classification Model")
        st.write("Predicts loan approval")

elif page == "Dataset":
    st.title("📁 Dataset")
    df = pd.read_csv("loan_approval_dataset.csv")
    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    st.dataframe(df.head(10))

elif page == "EDA":
    st.title("📊 Data Analysis")
    df = pd.read_csv("loan_approval_dataset.csv")
    st.write(df.describe())

elif page == "Model Training":
    st.title("🤖 Model Training")
    df = pd.read_csv("loan_approval_dataset.csv")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Regression")
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric) > 1:
            X = df[numeric[:-1]].fillna(0)
            y = df[numeric[-1]].fillna(0)
            Xt, Xte, yt, yte = train_test_split(X, y, test_size=0.2)
            m = RandomForestRegressor(n_estimators=50)
            m.fit(Xt, yt)
            r2 = r2_score(yte, m.predict(Xte))
            st.metric("R² Score", f"{r2:.4f}")
    
    with col2:
        st.subheader("Classification")
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric) > 1:
            X = df[numeric[:-1]].fillna(0)
            y = df[numeric[-1]].fillna(0)
            Xt, Xte, yt, yte = train_test_split(X, y, test_size=0.2)
            m = RandomForestClassifier(n_estimators=50)
            m.fit(Xt, yt)
            acc = accuracy_score(yte, m.predict(Xte))
            st.metric("Accuracy", f"{acc:.4f}")

elif page == "Predictions":
    st.title("🔮 Predictions")
    st.write("Ready for predictions")