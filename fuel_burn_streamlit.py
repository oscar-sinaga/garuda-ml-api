#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# =====================
# Konfigurasi
# =====================

API_URL = "http://localhost:8600"
EXCEL_PATH = "05. Database RP May 2025 - AC REGISTER.xlsx"
SHEET_NAME = "Raw"

RENAME_MAP = {
    "AIRCRAFT TYPE": "AIRCRAFT_TYPE",
    "AC REG": "AC_REG",
    "SERVICE TYPE": "SERVICE_TYPE",
    "FLIGHT ROUTE": "FLIGHT_ROUTE",
    "CARGO CARRIED": "CARGO_CARRIED",
    "FREIGHT CARRIED": "FREIGHT_CARRIED",
    "ASK (000)": "ASK_000",
    "ATK (000)": "ATK_000",
    "ATK PASSENGER (000)": "ATK_PASSENGER_000",
    "ASK (000) Y CLASS": "ASK_000_Y_CLASS",
    "ASK (000) C CLASS": "ASK_000_C_CLASS",
    "RTK (000)": "RTK_000",
    "RPK (000)": "RPK_000",
    "RPK (000) Y CLASS": "RPK_000_Y_CLASS",
    "RTK PASSENGER (000)": "RTK_PASSENGER_000",
    "ADMINISTRATION HO": "ADMINISTRATION_HO",
    "FUEL BURN (IN LITER)": "FUEL_BURN_LITER",
}

SELECTED_FEATURES = [
    "ROUNDTRIPROUTE",
    "AIRCRAFT_TYPE",
    "AC_REG",
    "SERVICE_TYPE",
    "FLIGHT_ROUTE",
    "CARGO_CARRIED",
    "FREIGHT_CARRIED",
    "ASK_000",
    "ATK_000",
    "ATK_PASSENGER_000",
    "ASK_000_Y_CLASS",
    "ASK_000_C_CLASS",
    "RTK_000",
    "RPK_000",
    "RPK_000_Y_CLASS",
    "RTK_PASSENGER_000",
    "ADMINISTRATION_HO",
]

CATEGORICAL_COLS = [
    "ROUNDTRIPROUTE",
    "AIRCRAFT_TYPE",
    "AC_REG",
    "SERVICE_TYPE",
    "FLIGHT_ROUTE",
]
NUMERIC_COLS = [c for c in SELECTED_FEATURES if c not in CATEGORICAL_COLS]


# =====================
# Helper
# =====================

@st.cache_data
def load_sample_data():
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, skiprows=1)
    df = df.iloc[:, 1:]
    df = df.rename(columns=RENAME_MAP)

    cols_needed = SELECTED_FEATURES
    df = df[cols_needed].dropna()

    return df


# =====================
# UI
# =====================

st.set_page_config(page_title="Fuel Burn Predictor", layout="wide")

st.title("✈️ Fuel Burn Predictor (XGBoost + API)")

menu = st.sidebar.radio("Menu", ["Prediksi", "Train Model"])


if menu == "Train Model":
    st.header("🔁 Train / Retrain Model")

    st.write(
        "API akan membaca file Excel, melatih model XGBoost, "
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

    df_sample = load_sample_data()
    sample_row = df_sample.sample(1, random_state=1).iloc[0]

    st.caption("Default value diisi dari salah satu contoh flight di dataset.")

    # =====================
    # Input fitur
    # =====================
    with st.form("predict_form"):
        col_cat1, col_cat2 = st.columns(2)
        col_cat3, col_cat4 = st.columns(2)
        col_cat5, _ = st.columns(2)

        # Opsi untuk categorical dari data
        rt_options = sorted(df_sample["ROUNDTRIPROUTE"].dropna().unique())
        atype_options = sorted(df_sample["AIRCRAFT_TYPE"].dropna().unique())
        ac_reg_options = sorted(df_sample["AC_REG"].dropna().unique())
        stype_options = sorted(df_sample["SERVICE_TYPE"].dropna().unique())
        froute_options = sorted(df_sample["FLIGHT_ROUTE"].dropna().unique())

        def default_index(options, value):
            try:
                return list(options).index(value)
            except ValueError:
                return 0

        ROUNDTRIPROUTE = col_cat1.selectbox(
            "ROUNDTRIPROUTE",
            rt_options,
            index=default_index(rt_options, sample_row["ROUNDTRIPROUTE"]),
        )
        AIRCRAFT_TYPE = col_cat2.selectbox(
            "AIRCRAFT_TYPE",
            atype_options,
            index=default_index(atype_options, sample_row["AIRCRAFT_TYPE"]),
        )
        AC_REG = col_cat3.selectbox(
            "AC_REG",
            ac_reg_options,
            index=default_index(ac_reg_options, sample_row["AC_REG"]),
        )
        SERVICE_TYPE = col_cat4.selectbox(
            "SERVICE_TYPE",
            stype_options,
            index=default_index(stype_options, sample_row["SERVICE_TYPE"]),
        )
        FLIGHT_ROUTE = col_cat5.selectbox(
            "FLIGHT_ROUTE",
            froute_options,
            index=default_index(froute_options, sample_row["FLIGHT_ROUTE"]),
        )

        st.markdown("---")
        st.subheader("Fitur Numerik")

        num_cols1, num_cols2, num_cols3 = st.columns(3)

        CARGO_CARRIED = num_cols1.number_input(
            "CARGO_CARRIED", value=float(sample_row["CARGO_CARRIED"])
        )
        FREIGHT_CARRIED = num_cols2.number_input(
            "FREIGHT_CARRIED", value=float(sample_row["FREIGHT_CARRIED"])
        )
        ASK_000 = num_cols3.number_input(
            "ASK_000", value=float(sample_row["ASK_000"])
        )

        ATK_000 = num_cols1.number_input(
            "ATK_000", value=float(sample_row["ATK_000"])
        )
        ATK_PASSENGER_000 = num_cols2.number_input(
            "ATK_PASSENGER_000", value=float(sample_row["ATK_PASSENGER_000"])
        )
        ASK_000_Y_CLASS = num_cols3.number_input(
            "ASK_000_Y_CLASS", value=float(sample_row["ASK_000_Y_CLASS"])
        )

        ASK_000_C_CLASS = num_cols1.number_input(
            "ASK_000_C_CLASS", value=float(sample_row["ASK_000_C_CLASS"])
        )
        RTK_000 = num_cols2.number_input(
            "RTK_000", value=float(sample_row["RTK_000"])
        )
        RPK_000 = num_cols3.number_input(
            "RPK_000", value=float(sample_row["RPK_000"])
        )

        RPK_000_Y_CLASS = num_cols1.number_input(
            "RPK_000_Y_CLASS", value=float(sample_row["RPK_000_Y_CLASS"])
        )
        RTK_PASSENGER_000 = num_cols2.number_input(
            "RTK_PASSENGER_000", value=float(sample_row["RTK_PASSENGER_000"])
        )
        ADMINISTRATION_HO = num_cols3.number_input(
            "ADMINISTRATION_HO", value=float(sample_row["ADMINISTRATION_HO"])
        )

        submitted = st.form_submit_button("🔮 Prediksi Fuel Burn")

    if submitted:
        record = {
            "ROUNDTRIPROUTE": ROUNDTRIPROUTE,
            "AIRCRAFT_TYPE": AIRCRAFT_TYPE,
            "AC_REG": AC_REG,
            "SERVICE_TYPE": SERVICE_TYPE,
            "FLIGHT_ROUTE": FLIGHT_ROUTE,
            "CARGO_CARRIED": CARGO_CARRIED,
            "FREIGHT_CARRIED": FREIGHT_CARRIED,
            "ASK_000": ASK_000,
            "ATK_000": ATK_000,
            "ATK_PASSENGER_000": ATK_PASSENGER_000,
            "ASK_000_Y_CLASS": ASK_000_Y_CLASS,
            "ASK_000_C_CLASS": ASK_000_C_CLASS,
            "RTK_000": RTK_000,
            "RPK_000": RPK_000,
            "RPK_000_Y_CLASS": RPK_000_Y_CLASS,
            "RTK_PASSENGER_000": RTK_PASSENGER_000,
            "ADMINISTRATION_HO": ADMINISTRATION_HO,
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
                st.metric("Perkiraan Fuel Burn (liter)", f"{pred:,.2f}")
                st.json(data)
            else:
                st.error(f"Error {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"Gagal menghubungi API: {e}")
