import hashlib
import os
import secrets
import sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import psycopg2
import streamlit as st
from psycopg2.extras import RealDictCursor


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "users.db"
DATABASE_URL = os.getenv("DATABASE_URL")


@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "knn_heart.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    expected_columns = joblib.load(BASE_DIR / "columns.pkl")
    return model, scaler, expected_columns


def get_connection():
    if DATABASE_URL:
        # PostgreSQL (Render)
        return psycopg2.connect(DATABASE_URL)
    else:
        # SQLite (Local)
        return sqlite3.connect(DB_PATH)


def get_placeholder():
    return "%s" if DATABASE_URL else "?"


def get_id_type():
    return "SERIAL PRIMARY KEY" if DATABASE_URL else "INTEGER PRIMARY KEY AUTOINCREMENT"


def get_integrity_error():
    return psycopg2.IntegrityError if DATABASE_URL else sqlite3.IntegrityError


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    id_type = get_id_type()
    
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS users (
            id {id_type},
            full_name TEXT NOT NULL,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id {id_type},
            user_id INTEGER NOT NULL,
            age INTEGER NOT NULL,
            resting_bp INTEGER NOT NULL,
            cholesterol INTEGER NOT NULL,
            max_hr INTEGER NOT NULL,
            prediction_result TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """
    )
    conn.commit()
    conn.close()


def hash_password(password):
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100000,
    ).hex()
    return f"{salt}${digest}"


def verify_password(password, stored_value):
    salt, saved_digest = stored_value.split("$", maxsplit=1)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100000,
    ).hex()
    return digest == saved_digest


def create_user(full_name, username, password):
    conn = get_connection()
    cur = conn.cursor()
    placeholder = get_placeholder()
    IntegrityError = get_integrity_error()
    
    try:
        cur.execute(
            f"""
            INSERT INTO users (full_name, username, password_hash, created_at)
            VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder})
            """,
            (
                full_name.strip(),
                username.strip().lower(),
                hash_password(password),
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
        return True, "Account created successfully. Please log in."
    except IntegrityError:
        return False, "That username already exists."
    finally:
        conn.close()


def authenticate_user(username, password):
    conn = get_connection()
    cur = conn.cursor()
    placeholder = get_placeholder()
    
    cur.execute(
        f"""
        SELECT id, full_name, username, password_hash
        FROM users
        WHERE username = {placeholder}
        """,
        (username.strip().lower(),),
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    user_id, full_name, saved_username, password_hash = row
    if verify_password(password, password_hash):
        return {
            "id": user_id,
            "full_name": full_name,
            "username": saved_username,
        }
    return None


def save_prediction(user_id, age, resting_bp, cholesterol, max_hr, prediction_result):
    conn = get_connection()
    cur = conn.cursor()
    placeholder = get_placeholder()
    
    cur.execute(
        f"""
        INSERT INTO prediction_history (
            user_id, age, resting_bp, cholesterol, max_hr, prediction_result, created_at
        )
        VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
        """,
        (
            user_id,
            age,
            resting_bp,
            cholesterol,
            max_hr,
            prediction_result,
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def get_user_prediction_history(user_id, limit=None):
    conn = get_connection()
    cur = conn.cursor()
    placeholder = get_placeholder()
    
    query = f"""
        SELECT age, resting_bp, cholesterol, max_hr, prediction_result, created_at
        FROM prediction_history
        WHERE user_id = {placeholder}
        ORDER BY created_at DESC
    """
    params = [user_id]
    if limit is not None:
        query += f" LIMIT {placeholder}"
        params.append(limit)
    
    cur.execute(query, tuple(params))
    rows = cur.fetchall()
    conn.close()

    return pd.DataFrame(
        rows,
        columns=[
            "Age",
            "Resting BP",
            "Cholesterol",
            "Max HR",
            "Result",
            "Saved At",
        ],
    )


def delete_user_prediction_history(user_id):
    conn = get_connection()
    cur = conn.cursor()
    placeholder = get_placeholder()
    
    cur.execute(
        f"DELETE FROM prediction_history WHERE user_id = {placeholder}",
        (user_id,),
    )
    conn.commit()
    conn.close()


def render_recommendations(prediction_result, age, resting_bp, cholesterol):
    st.subheader("💡 Personalized Health Recommendations")
    
    if "High Risk" in prediction_result:
        st.warning("Your recent prediction indicates a high risk. Here are some actionable steps:")
        
        recs = []
        if resting_bp > 140:
            recs.append("- **Blood Pressure Control:** Your BP is high. Reduce salt intake, exercise regularly, and consult a doctor about medication.")
        if cholesterol > 240:
            recs.append("- **Cholesterol Management:** Focus on a heart-healthy diet rich in fiber and low in saturated fats. Consider Omega-3 supplements.")
        if age > 50:
            recs.append("- **Age-Related Monitoring:** Regular checkups are crucial at your age. Monitor your heart rate during exercise.")
        
        recs.append("- **Lifestyle Changes:** Avoid smoking, limit alcohol, and practice stress-management techniques like meditation.")
        
        for rec in recs:
            st.write(rec)
            
        st.info("⚠️ *Disclaimer: These are general suggestions. Please consult a medical professional for personalized advice.*")
    else:
        st.success("Great job! Your current risk is low. Maintain your healthy habits.")
        st.write("- **Consistency is Key:** Keep up with your balanced diet and regular physical activity.")
        st.write("- **Regular Screening:** Even with low risk, yearly checkups are recommended.")


def render_visualizations(history_df):
    if history_df.empty:
        return

    st.subheader("📈 Health Trends Over Time")
    
    # Sort by Saved At for proper time series
    history_df["Saved At"] = pd.to_datetime(history_df["Saved At"])
    viz_df = history_df.sort_values("Saved At")

    # Metrics selection for comparison
    metrics = ["Resting BP", "Cholesterol", "Max HR"]
    
    # Using a modern color palette
    color_map = {
        "Resting BP": "#4F46E5",  # Indigo
        "Cholesterol": "#10B981", # Teal
        "Max HR": "#F59E0B"        # Amber
    }
    
    fig = px.line(
        viz_df, 
        x="Saved At", 
        y=metrics, 
        title="Vital Metrics Comparison Over Time",
        labels={"value": "Metric Value", "variable": "Health Metric"},
        markers=True,
        color_discrete_map=color_map
    )
    
    fig.update_layout(
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#1E293B"),  # Explicitly set font for all elements
        title_font=dict(size=20, color="#1E293B"),
        legend=dict(font=dict(color="#1E293B")),
        xaxis=dict(
            title_font=dict(color="#1E293B"),
            tickfont=dict(color="#1E293B"),
            gridcolor="#E2E8F0"
        ),
        yaxis=dict(
            title_font=dict(color="#1E293B"),
            tickfont=dict(color="#1E293B"),
            gridcolor="#E2E8F0"
        ),
        legend_title_text=''
    )
    st.plotly_chart(fig, use_container_width=True)

    # Risk progression
    st.markdown("**Risk History Visualization**")
    fig_risk = px.scatter(
        viz_df,
        x="Saved At",
        y="Result",
        color="Result",
        color_discrete_map={
            "High Risk of Heart Disease": "#EF4444",  # Rose/Red
            "Low Risk of Heart Disease": "#10B981"    # Teal/Green
        },
        title="Prediction Risk Progression"
    )
    fig_risk.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#1E293B"),
        title_font=dict(color="#1E293B"),
        xaxis=dict(
            title_font=dict(color="#1E293B"),
            tickfont=dict(color="#1E293B"),
            gridcolor="#E2E8F0"
        ),
        yaxis=dict(
            title_font=dict(color="#1E293B"),
            tickfont=dict(color="#1E293B"),
            gridcolor="#E2E8F0"
        )
    )
    st.plotly_chart(fig_risk, use_container_width=True)


def render_auth_page():
    # Center the login form
    _, col, _ = st.columns([1, 2, 1])
    
    with col:
        st.title("🔐 Heart Risk Pro")
        st.markdown("### Welcome Back")
        st.write("Please log in or create an account to continue.")

        login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

        with login_tab:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Log In")

            if submitted:
                user = authenticate_user(username, password)
                if user:
                    st.session_state.user = user
                    st.success("Login successful.")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        with signup_tab:
            with st.form("signup_form"):
                full_name = st.text_input("Full Name")
                username = st.text_input("Choose a Username")
                password = st.text_input("Choose a Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                submitted = st.form_submit_button("Create Account")

            if submitted:
                if not full_name or not username or not password:
                    st.error("Please fill in all fields.")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters long.")
                elif password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    created, message = create_user(full_name, username, password)
                    if created:
                        st.success(message)
                    else:
                        st.error(message)


def render_prediction_page(model, scaler, expected_columns):
    # Sidebar Profile Section
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.user['full_name']}")
        st.write(f"@{st.session_state.user['username']}")
        if st.button("Logout", use_container_width=True):
            st.session_state.pop("user", None)
            st.rerun()
        st.divider()

    st.title("💓 Heart Risk Pro")
    st.write("Enter patient details below to analyze heart disease risk.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Basic Information")
            age = st.number_input("Age", 1, 120, 30)
            sex = st.selectbox("Sex", ["Male", "Female"])
            resting_bp = st.number_input("Resting Blood Pressure", 50, 250, 120)
            cholesterol = st.number_input("Cholesterol", 0, 600, 200)

        with col2:
            st.markdown("### 🏥 Clinical Tests")
            fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])
            max_hr = st.number_input("Max Heart Rate", 60, 250, 150)
            oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 10.0, 1.0)
            chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
            resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])

        st.markdown("### 🏃 Activity")
        exercise_col, slope_col = st.columns(2)
        with exercise_col:
            exercise_angina = st.selectbox("Exercise Induced Angina?", ["No", "Yes"])
        with slope_col:
            st_slope = st.selectbox("ST Slope", ["Flat", "Up", "Down"])

        submitted = st.form_submit_button("Predict Risk Level")

    if submitted:
        input_data = {
            "Age": age,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": 1 if fasting_bs == "Yes" else 0,
            "MaxHR": max_hr,
            "Oldpeak": oldpeak,
            "Sex_M": 1 if sex == "Male" else 0,
            "ChestPainType_ATA": 1 if chest_pain == "ATA" else 0,
            "ChestPainType_NAP": 1 if chest_pain == "NAP" else 0,
            "ChestPainType_TA": 1 if chest_pain == "TA" else 0,
            "RestingECG_Normal": 1 if resting_ecg == "Normal" else 0,
            "RestingECG_ST": 1 if resting_ecg == "ST" else 0,
            "ExerciseAngina_Y": 1 if exercise_angina == "Yes" else 0,
            "ST_Slope_Flat": 1 if st_slope == "Flat" else 0,
            "ST_Slope_Up": 1 if st_slope == "Up" else 0,
        }

        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=expected_columns, fill_value=0)
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        prediction_result = (
            "High Risk of Heart Disease" if prediction[0] == 1 else "Low Risk of Heart Disease"
        )

        save_prediction(
            st.session_state.user["id"],
            age,
            resting_bp,
            cholesterol,
            max_hr,
            prediction_result,
        )

        if prediction[0] == 1:
            st.error("High Risk of Heart Disease")
        else:
            st.success("Low Risk of Heart Disease")

        # Show recommendations immediately after prediction
        render_recommendations(prediction_result, age, resting_bp, cholesterol)

    # Historical Visualization
    history_df = get_user_prediction_history(st.session_state.user["id"])
    if not history_df.empty:
        render_visualizations(history_df)

    st.subheader("Recent Prediction History")
    # Limit to latest 10 for table display
    table_history_df = get_user_prediction_history(st.session_state.user["id"], limit=10)
    
    action_col1, action_col2 = st.columns(2)
    with action_col1:
        st.download_button(
            "Download History CSV",
            data=history_df.to_csv(index=False),
            file_name=f"{st.session_state.user['username']}_prediction_history.csv",
            mime="text/csv",
            disabled=history_df.empty,
        )
    with action_col2:
        if st.button("Delete History", disabled=history_df.empty):
            delete_user_prediction_history(st.session_state.user["id"])
            st.success("Prediction history deleted.")
            st.rerun()

    if table_history_df.empty:
        st.info("No predictions saved yet.")
    else:
        st.dataframe(table_history_df, use_container_width=True)


def main():
    st.set_page_config(
        page_title="Heart Risk Pro",
        page_icon="💓",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for a more attractive UI
    st.markdown("""
        <style>
        /* Global body override */
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #F1F5F9 !important;
        }
        
        /* Main container padding and background */
        .block-container {
            background-color: #F1F5F9 !important;
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }
        
        /* Target EVERYTHING for background color */
        .stApp, .main, .stSidebar, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {
            background-color: #F1F5F9 !important;
        }

        /* Force light theme for ALL buttons including download buttons */
        .stButton>button, .stDownloadButton>button, [data-testid="stFormSubmitButton"]>button {
            background-color: #4F46E5 !important;
            color: white !important;
            border-radius: 10px !important;
            border: none !important;
            padding: 0.6rem 2.5rem !important;
            font-weight: 600 !important;
        }

        /* Specifically target the download button which often stays dark */
        .stDownloadButton > button {
            background-color: #4F46E5 !important;
            color: #FFFFFF !important;
        }
        
        /* Force light theme for dataframes and tables specifically */
        [data-testid="stTable"], [data-testid="stDataFrame"], .stTable, .stDataFrame, div[role="grid"], .glideDataEditor {
            background-color: #FFFFFF !important;
            color: #1E293B !important;
        }
        
        /* Force light theme for table cells and headers */
        [data-testid="stTable"] td, [data-testid="stTable"] th, [role="gridcell"], [role="columnheader"] {
            background-color: #FFFFFF !important;
            color: #1E293B !important;
        }

        /* Target specific Streamlit elements that might stay dark */
        div[data-testid="stVerticalBlock"] > div {
            background-color: transparent !important;
        }
        
        /* Headers and titles */
        h1, h2, h3, h4, h5, h6, p, span, label {
            color: #1E293B !important;
            font-family: 'Inter', 'Segoe UI', sans-serif !important;
        }
        
        /* Sidebar specific override */
        [data-testid="stSidebar"], [data-testid="stSidebarNav"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E2E8F0 !important;
        }

        /* Form containers */
        [data-testid="stForm"] {
            background-color: #FFFFFF !important;
            padding: 2.5rem !important;
            border-radius: 20px !important;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1) !important;
            border: 1px solid #E2E8F0 !important;
        }

        /* Target ALL input fields specifically */
        input, select, textarea, [data-baseweb="input"] {
            background-color: #FFFFFF !important;
            color: #1E293B !important;
            border-color: #E2E8F0 !important;
        }
        
        /* BaseWeb specific overrides (Streamlit uses BaseWeb) */
        [data-baseweb="input"] > div {
            background-color: #FFFFFF !important;
            color: #1E293B !important;
        }

        [data-baseweb="select"] > div {
            background-color: #FFFFFF !important;
            color: #1E293B !important;
        }

        /* Buttons */
        .stButton>button, [data-testid="stFormSubmitButton"]>button {
            background-color: #4F46E5 !important;
            color: white !important;
            border-radius: 10px !important;
            border: none !important;
            padding: 0.6rem 2.5rem !important;
            font-weight: 600 !important;
            width: 100% !important; /* Make button full width in login */
        }
        
        .stButton>button:hover {
            background-color: #4338CA;
            transform: translateY(-1px);
            box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.4);
        }
        
        /* Dataframes */
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #E2E8F0;
        }

        /* Recommendations card */
        .recommendation-card {
            background-color: #FFFFFF;
            padding: 1.5rem;
            border-left: 6px solid #4F46E5;
            border-radius: 8px;
            margin: 1.5rem 0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)

    init_db()
    model, scaler, expected_columns = load_artifacts()

    if "user" not in st.session_state:
        st.session_state.user = None

    if st.session_state.user:
        render_prediction_page(model, scaler, expected_columns)
    else:
        render_auth_page()


if __name__ == "__main__":
    main()
