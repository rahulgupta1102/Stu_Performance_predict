import streamlit as st
import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# ✅ MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="🎓 Student Performance Predictor",
    page_icon="📈",
    layout="centered"
)

# ------------------ LOAD DATA ------------------
csv_files = glob.glob("*.csv")

if len(csv_files) == 0:
    st.error("❌ No CSV file found. Place your dataset in the same folder as app.py.")
    st.stop()

file_name = csv_files[0]
st.write(f"📂 Using dataset file: **{file_name}**")

df = pd.read_csv(file_name)

# ------------------ DATA PREPROCESSING ------------------
if 'StudentID' in df.columns:
    df = df.drop('StudentID', axis=1)

le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

if 'GPA' not in df.columns:
    st.error("❌ Target column 'GPA' not found in dataset.")
    st.stop()

X = df.drop('GPA', axis=1)
y = df['GPA']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr = LinearRegression().fit(X_train, y_train)
dt = DecisionTreeRegressor(random_state=42, max_depth=5).fit(X_train, y_train)

# ------------------ STREAMLIT UI ------------------
st.title("🎓 Student Performance Predictor")
st.write("Predict a student's **GPA** based on their background, habits, and activities.")

st.sidebar.header("🧩 Enter Student Details")

input_data = {}
for col in X.columns:
    if str(df[col].dtype) == 'object' or len(df[col].unique()) <= 10:
        input_data[col] = st.sidebar.selectbox(col, sorted(df[col].unique()))
    else:
        min_val, max_val = float(df[col].min()), float(df[col].max())
        default_val = float(df[col].mean())
        input_data[col] = st.sidebar.slider(col, min_val, max_val, default_val)

sample = pd.DataFrame([input_data])

sample = sample.reindex(columns=X.columns, fill_value=0)
sample_scaled = scaler.transform(sample)

st.write("### 🔍 Choose Model for Prediction")
model_choice = st.radio("Select Model:", ("Linear Regression", "Decision Tree"))

if st.button("🎯 Predict GPA"):
    pred = lr.predict(sample_scaled)[0] if model_choice == "Linear Regression" else dt.predict(sample_scaled)[0]

    st.success(f"🎓 Predicted GPA: **{pred:.2f}**")

    if pred >= 3.5:
        st.balloons()
        st.info("Excellent performance! 🌟 Keep it up!")
    elif pred >= 2.5:
        st.warning("Good performance 👍 Some improvement possible.")
    else:
        st.error("Below average 😟 — focus on study time and attendance.")
