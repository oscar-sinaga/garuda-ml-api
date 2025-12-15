"""
Streamlit UI untuk Fuel Burn XGBoost API

Jalankan:
1) Pastikan API sudah jalan di http://localhost:8600
2) pip install streamlit pandas requests openpyxl
3) streamlit run fuel_burn_streamlit.py
"""

import requests
import streamlit as st
import pandas as pd
import random

# =====================
# Konfigurasi
# =====================

API_URL = "http://localhost:8600"
EXCEL_PATH = "../data/05. Database RP May 2025 - AC REGISTER.xlsx"
SHEET_NAME = "Raw"

RENAME_MAP = {
    "FLIGHT KILOMETERS": "FLIGHT_KILOMETERS",
    "SEAT OFFERED": "SEAT_OFFERED",
    "PASSENGER CARRIED" : "PASSENGER_CARRIED",
    "PASSENGER COMMISSION": "PASSENGER_COMMISSION",
    "BLOCK HOURS": "BLOCK_HOURS",
    "FUEL AIRCRAFT": "FUEL_AIRCRAFT",

    "RPK (000) C CLASS": "RPK_000_C_CLASS",
    "RTK (000)": "RTK_000",
    "RPK (000)": "RPK_000",
    "RTK PASSENGER (000)": "RTK_PASSENGER_000",
    "RPK (000) Y CLASS": "RPK_000_Y_CLASS",
    "ASK (000) C CLASS": "ASK_000_C_CLASS",
    "PASSENGER CARRIED C CLASS": "PASSENGER_CARRIED_C_CLASS"
}

SELECTED_FEATURES = [
    "RPK_000_C_CLASS",
    "RTK_000",
    "RPK_000",
    "RTK_PASSENGER_000",
    "RPK_000_Y_CLASS",
    "ASK_000_C_CLASS",
    "PASSENGER_CARRIED_C_CLASS"
]

# =====================
# Helper
# =====================

@st.cache_data
def load_sample_data(nrows=1000):
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, skiprows=1, nrows=nrows)
    df = df.iloc[:, 1:]
    df = df.rename(columns=RENAME_MAP)

    cols_needed = SELECTED_FEATURES + ["PASSENGER_COMMISSION"]
    df = df[cols_needed].dropna()

    return df

# =====================
# UI
# =====================

st.set_page_config(page_title="Passenger Commission Predictor", layout="wide")

st.title("✈️ Passenger Commission Predictor (LightGBM + API)")

menu = st.sidebar.radio("Menu", ["Prediksi", "Train Model"])

if menu == "Train Model":
    st.header("🔁 Train / Retrain Model")

    st.write(
        "API akan membaca file Excel, melatih model LightGBM, "
        "menyimpan model ke disk, dan mengembalikan MAPE & RMSE."
    )

    if st.button("Train sekarang"):
        try:
            with st.spinner("Training model di server API..."):
                resp = requests.post(f"{API_URL}/train")
            if resp.status_code == 200:
                data = resp.json()
                st.success("Training selesai ✅")
                st.json(data)
                st.metric("MAPE (%)", f"{data['mape_percent']:.2f}")
                st.metric("RMSE (liter)", f"{data['rmse']:.2f}")
            else:
                st.error(f"Error {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"Gagal menghubungi API: {e}")

elif menu == "Prediksi":
    st.header("📈 Prediksi Fuel Burn (Liter)")

    df_sample = load_sample_data(1000)
    sample_row = df_sample.sample(1, random_state=1).iloc[0]
    
    actual = sample_row["PASSENGER_COMMISSION"]

    st.caption("Default value diisi dari salah satu contoh flight di dataset.")

    # =====================
    # Input fitur
    # =====================
    with st.form("predict_form"):
        col_cat1, col_cat2 = st.columns(2)
        col_cat3, col_cat4 = st.columns(2)
        col_cat5, _ = st.columns(2)

        st.subheader("Fitur Numerik")

        num_cols1, num_cols2, num_cols3 = st.columns(3)

        RPK_000_C_CLASS = num_cols1.number_input(
            "RPK_000_C_CLASS", value=float(sample_row["RPK_000_C_CLASS"])
        )

        RTK_000 = num_cols2.number_input(
            "RTK_000", value=float(sample_row["RTK_000"])
        )

        RPK_000 = num_cols3.number_input(
            "RPK_000", value=float(sample_row["RPK_000"])
        )

        RTK_PASSENGER_000 = num_cols1.number_input(
            "RTK_PASSENGER_000", value=float(sample_row["RTK_PASSENGER_000"])
        )

        RPK_000_Y_CLASS = num_cols2.number_input(
            "RPK_000_Y_CLASS", value=float(sample_row["RPK_000_Y_CLASS"])
        )
        
        ASK_000_C_CLASS = num_cols3.number_input(
            "ASK_000_C_CLASS", value=float(sample_row["ASK_000_C_CLASS"])
        )
        
        PASSENGER_CARRIED_C_CLASS = num_cols1.number_input(
            "PASSENGER_CARRIED_C_CLASS", value=float(sample_row["PASSENGER_CARRIED_C_CLASS"])
        )

        submitted = st.form_submit_button("🔮 Prediksi Passenger Commission")

    if submitted:
        record = {
            "RPK_000_C_CLASS" : RPK_000_C_CLASS,
            "RTK_000" : RTK_000,
            "RPK_000": RPK_000,
            "RTK_PASSENGER_000":RTK_PASSENGER_000,
            "RPK_000_Y_CLASS":RPK_000_Y_CLASS,
            "ASK_000_C_CLASS":ASK_000_C_CLASS,
            "PASSENGER_CARRIED_C_CLASS": PASSENGER_CARRIED_C_CLASS
        }

        try:
            with st.spinner("Meminta prediksi ke API..."):
                resp = requests.post(
                    f"{API_URL}/predict", json={"records": [record]}
                )
            if resp.status_code == 200:
                data = resp.json()
                pred = data["predictions"][0]
                st.success("Prediksi berhasil ✅")
                st.metric("Perkiraan Passenger Commission", f"{pred:,.2f}")
                st.json(data)
                st.text( f"{actual:,.2f}")
            else:
                st.error(f"Error {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"Gagal menghubungi API: {e}")
