import hashlib
import os
import secrets
import sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
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


def render_auth_page():
    st.title("Heart Disease Prediction Login")
    st.write("Create an account or log in to access the prediction tool.")

    login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")

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
    st.title("Heart Disease Prediction")
    st.write(f"Signed in as **{st.session_state.user['full_name']}**")
    st.write("Enter patient details below:")

    if st.sidebar.button("Logout"):
        st.session_state.pop("user", None)
        st.rerun()

    with st.form("prediction_form"):
        age = st.number_input("Age", 1, 120, 30)
        resting_bp = st.number_input("Resting Blood Pressure", 50, 250, 120)
        cholesterol = st.number_input("Cholesterol", 0, 600, 200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])
        max_hr = st.number_input("Max Heart Rate", 60, 250, 150)
        oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 10.0, 1.0)

        sex = st.selectbox("Sex", ["Male", "Female"])
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        exercise_angina = st.selectbox("Exercise Induced Angina?", ["No", "Yes"])
        st_slope = st.selectbox("ST Slope", ["Flat", "Up", "Down"])

        submitted = st.form_submit_button("Predict")

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

    st.subheader("Recent Prediction History")
    history_df = get_user_prediction_history(st.session_state.user["id"], limit=10)
    export_history_df = get_user_prediction_history(st.session_state.user["id"])

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        st.download_button(
            "Download History CSV",
            data=export_history_df.to_csv(index=False),
            file_name=f"{st.session_state.user['username']}_prediction_history.csv",
            mime="text/csv",
            disabled=export_history_df.empty,
        )
    with action_col2:
        if st.button("Delete History", disabled=history_df.empty):
            delete_user_prediction_history(st.session_state.user["id"])
            st.success("Prediction history deleted.")
            st.rerun()

    if history_df.empty:
        st.info("No predictions saved yet.")
    else:
        st.dataframe(history_df, use_container_width=True)


def main():
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
