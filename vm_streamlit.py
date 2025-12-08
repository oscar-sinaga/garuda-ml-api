#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import requests

# ============================
# KONFIGURASI API
# ============================
API_PREDICT = "http://localhost:8700/predict_vm"
API_TRAIN = "http://localhost:8700/train_vm"

# ============================
# KONFIGURASI FILE DATA (untuk dropdown)
# ============================
VM_EXCEL_PATH = "05. Database RP May 2025 - AC REGISTER.xlsx"  # GANTI SESUAI FILE KAMU
VM_SHEET = "Raw"

st.set_page_config(page_title="Variable Maintenance Predictor", layout="wide")

# ============================
# LOAD DATA FOR AUTOCOMPLETE
# ============================
@st.cache_data
def load_vm_data():
    df = pd.read_excel(VM_EXCEL_PATH, sheet_name=VM_SHEET,skiprows=1)
    df = df.iloc[:, 1:]  # Drop kolom index pertama seperti notebook
    df = df.fillna("")

    return df


st.title("🛠️ Variable Maintenance Prediction System (XGBoost)")

menu = st.sidebar.radio("Menu", ["Prediksi VM", "Train / Retrain Model"])


# ============================================================
# MENU 1 — PREDIKSI VARIABLE MAINTENANCE DENGAN CONDITIONAL FILTER
# ============================================================

if menu == "Prediksi VM":

    st.header("🔮 Prediksi Variable Maintenance")
    st.write("Dropdown kategori otomatis menyesuaikan pilihan sebelumnya.")

    df = load_vm_data()


    # Unique categories
    opt_roundtrip = sorted(df['ROUNDTRIPROUTE'].astype(str).unique())
    opt_aircraft_type = sorted(df['AIRCRAFT TYPE'].astype(str).unique())
    opt_service_type = sorted(df['SERVICE TYPE'].astype(str).unique())
    opt_acreg = sorted(df['AC REG'].astype(str).unique())
    opt_route = sorted(df['FLIGHT ROUTE'].astype(str).unique())
    opt_periode = sorted(df['PERIODE'].astype(str).unique())

    # Dataset untuk filter
    # df = df.copy()

    sample_row = df.sample(1, random_state=1).iloc[0]

    # ============================
    # LANGKAH 1 — PILIH AC REG
    # ============================
    AC_REG = st.selectbox(
        "Pilih AC REG",
        sorted(df['AC REG'].astype(str).unique())
    )

    df = df[df['AC REG'] == AC_REG]

    # ============================
    # LANGKAH 2 — FILTER PERIODE
    # ============================
    PERIODE = st.selectbox(
        "Pilih PERIODE",
        sorted(df['PERIODE'].astype(str).unique())
    )

    df = df[df['PERIODE'] == PERIODE]

    # ============================
    # LANGKAH 3 — FILTER AIRCRAFT TYPE
    # ============================
    AIRCRAFT_TYPE = st.selectbox(
        "AIRCRAFT TYPE",
        sorted(df['AIRCRAFT TYPE'].astype(str).unique())
    )

    df = df[df['AIRCRAFT TYPE'] == AIRCRAFT_TYPE]

    # ============================
    # LANGKAH 4 — FILTER SERVICE TYPE
    # ============================
    SERVICE_TYPE = st.selectbox(
        "SERVICE TYPE",
        sorted(df['SERVICE TYPE'].astype(str).unique())
    )

    df = df[df['SERVICE TYPE'] == SERVICE_TYPE]

    # ============================
    # LANGKAH 5 — FILTER ROUNDTRIPROUTE
    # ============================
    ROUNDTRIPROUTE = st.selectbox(
        "ROUNDTRIPROUTE",
        sorted(df['ROUNDTRIPROUTE'].astype(str).unique())
    )

    df = df[df['ROUNDTRIPROUTE'] == ROUNDTRIPROUTE]

    # ============================
    # LANGKAH 6 — FILTER FLIGHT ROUTE
    # ============================
    FLIGHT_ROUTE = st.selectbox(
        "FLIGHT ROUTE",
        sorted(df['FLIGHT ROUTE'].astype(str).unique())
    )

    # =====================================
    # NUMERIC FEATURES
    # =====================================

    st.markdown("### ✏️ Masukkan Fitur Numerik")

    c1, c2, c3 = st.columns(3)

    BLOCK_HOURS = c1.number_input("BLOCK HOURS", min_value=0.0,value=float(sample_row["BLOCK HOURS"]))
    FLIGHT_HOURS = c2.number_input("FLIGHT HOURS", min_value=0.0,value=float(sample_row["FLIGHT HOURS"]))
    FUEL_BURN_IN_LITER = c3.number_input("FUEL BURN (IN LITER)", min_value=0.0,value=float(sample_row["FUEL BURN (IN LITER)"]))

    ASK_000 = c1.number_input("ASK (000)", min_value=0.0,value=float(sample_row["ASK (000)"]))
    ATK_PASSENGER_000 = c2.number_input("ATK PASSENGER (000)", min_value=0.0,value=float(sample_row["ATK PASSENGER (000)"]))
    ATK_000 = c3.number_input("ATK (000)", min_value=0.0,value=float(sample_row["ATK (000)"]))

    LEASE_AIRCRAFT = c1.number_input("LEASE AIRCRAFT", min_value=0.0,value=float(sample_row["LEASE AIRCRAFT"]))
    CABIN_CREW_TRAVEL = c2.number_input("CABIN CREW TRAVEL", min_value=0.0,value=float(sample_row["CABIN CREW TRAVEL"]))
    FUEL_AIRCRAFT = c3.number_input("FUEL AIRCRAFT", min_value=0.0,value=float(sample_row["FUEL AIRCRAFT"]))

    COCKPIT_CREW_TRAVEL = c1.number_input("COCKPIT CREW TRAVEL", min_value=0.0,value=float(sample_row["COCKPIT CREW TRAVEL"]))
    CABIN_CREW_PERSON = c2.number_input("CABIN CREW PERSON", min_value=0.0,value=float(sample_row["CABIN CREW PERSON"]))

    # =====================================
    # SUBMIT BUTTON
    # =====================================

    if st.button("🔮 Prediksi VM Sekarang"):

        record = {
            "BLOCK_HOURS": BLOCK_HOURS,
            "FLIGHT_HOURS": FLIGHT_HOURS,
            "FUEL_BURN_IN_LITER": FUEL_BURN_IN_LITER,
            "ASK_000": ASK_000,
            "ATK_PASSENGER_000": ATK_PASSENGER_000,
            "ATK_000": ATK_000,
            "LEASE_AIRCRAFT": LEASE_AIRCRAFT,
            "CABIN_CREW_TRAVEL": CABIN_CREW_TRAVEL,
            "FUEL_AIRCRAFT": FUEL_AIRCRAFT,
            "COCKPIT_CREW_TRAVEL": COCKPIT_CREW_TRAVEL,
            "CABIN_CREW_PERSON": CABIN_CREW_PERSON,

            "ROUNDTRIPROUTE": ROUNDTRIPROUTE,
            "AIRCRAFT_TYPE": AIRCRAFT_TYPE,
            "SERVICE_TYPE": SERVICE_TYPE,
            "AC_REG": AC_REG,
            "FLIGHT_ROUTE": FLIGHT_ROUTE,
            "PERIODE": PERIODE
        }

        try:
            with st.spinner("Menghubungi API..."):
                resp = requests.post(API_PREDICT, json={"records": [record]})

            if resp.status_code == 200:
                pred = resp.json()['predictions'][0]
                st.success("Prediksi Berhasil 🎉")
                st.metric("Prediksi Variable Maintenance ($)", f"{pred:,.2f}")
                st.json(resp.json())
            else:
                st.error(f"Error {resp.status_code}: {resp.text}")

        except Exception as e:
            st.error(f"Gagal menghubungi API: {e}")




# ============================================================================
# MENU 2 — TRAIN / RETRAIN MODEL
# ============================================================================

elif menu == "Train / Retrain Model":

    st.header("🔁 Latih / Retrain Model Variable Maintenance")
    st.write("""
        Endpoint ini akan membaca dataset yang sudah ditentukan di API (`VM_EXCEL_PATH`),
        melakukan preprocessing, training ulang XGBoost, kemudian menyimpan model baru.
    """)

    st.warning("""
    ⚠️ Perhatian:
    - Proses training bisa memakan waktu (tergantung size dataset).
    - Model lama akan di-*overwrite* oleh model baru.
    """)

    if st.button("🚀 Train Model Sekarang"):
        try:
            with st.spinner("Sedang melatih model di server API..."):
                resp = requests.post(API_TRAIN)

            if resp.status_code == 200:
                st.success("Training model selesai! 🎉")

                data = resp.json()

                st.metric("MAPE (%)", f"{data['mape_percent']:.2f}")
                st.metric("RMSE", f"{data['rmse']:.4f}")
                st.metric("Jumlah Train", data["n_train"])
                st.metric("Jumlah Test", data["n_test"])

                st.subheader("📄 Detail Response")
                st.json(data)

            else:
                st.error(f"Training gagal — Error {resp.status_code}")
                st.text(resp.text)

        except Exception as e:
            st.error(f"Gagal menghubungi API: {e}")
