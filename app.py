import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- PAGE SETUP ---
st.set_page_config(page_title="Sharjah Traffic Model", layout="centered")
st.title("🚗 Sharjah-Dubai Traffic Predictor")
st.markdown("**Math 101 Project | Linear Algebra & Least Squares Approximation**")


# --- MATH ENGINE (Runs once and caches) ---
@st.cache_data
def train_model():
    # 1. Load data
    df = pd.read_csv('sharjah_congestion.csv')
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    y = df[days].values.flatten('F')  # Unroll to 168x1 vector

    # 2. Build Design Matrix X
    hours = np.tile(np.arange(24), 7)
    X = np.zeros((168, 9))
    X[:, 0] = 1  # Intercept (Monday Baseline)
    X[:, 1] = np.sin(2 * np.pi * hours / 24)
    X[:, 2] = np.cos(2 * np.pi * hours / 24)

    # Indicator variables (Tue-Sun, dropping Mon for Linear Independence)
    for i, day_idx in enumerate(range(1, 7)):
        X[day_idx * 24: (day_idx + 1) * 24, i + 3] = 1

    # 3. Solve Normal Equation: beta = (X^T X)^-1 X^T y
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return X, y, beta


X_matrix, y_actual, beta = train_model()

# --- USER INTERFACE ---
st.sidebar.header("Plan Your Commute")
selected_day = st.sidebar.selectbox("Select Day",
                                    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
selected_hour = st.sidebar.slider("Select Hour (24H)", 0, 23, 8)

# --- PREDICTION LOGIC ---
day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
day_idx = day_map[selected_day]

# Build the feature vector for the user's specific time
user_features = np.zeros(9)
user_features[0] = 1
user_features[1] = np.sin(2 * np.pi * selected_hour / 24)
user_features[2] = np.cos(2 * np.pi * selected_hour / 24)
if day_idx > 0:
    user_features[day_idx + 2] = 1  # Set the specific day indicator to 1

# Project it! (Dot product of features and weights)
predicted_congestion = np.dot(user_features, beta)

st.subheader(f"Predicted Congestion: {predicted_congestion:.1f}%")

if predicted_congestion > 70:
    st.error("🚨 High Congestion! Staggered scheduling recommended.")
elif predicted_congestion > 40:
    st.warning("⚠️ Moderate Traffic. Expect some delays.")
else:
    st.success("✅ Clear Roads. Optimal time to leave!")

# --- PROFESSOR ZIKKOS FLEX ZONE (The Math) ---
with st.expander("🧮 See the Mathematical Reasoning "):
    st.markdown(
        "Because the real-world traffic vector $y$ is not in the Column Space of our matrix $X$, the system $X\\beta = y$ is inconsistent. We find the **Orthogonal Projection** using the Normal Equation:")
    st.latex(r"\beta = (X^T X)^{-1} X^T y")
    st.markdown(
        "We dropped the 'Monday' indicator to prevent the **Dummy Variable Trap**, ensuring our column vectors are **Linearly Independent** and the matrix $X^T X$ is invertible.")
