#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Streamlit App
- Fuel Burn Model (Predict & Train)
- Variable Maintenance Model (Predict & Train)
Menggunakan 1 FastAPI backend (API gabungan)
"""

import streamlit as st
import pandas as pd
import requests

# ==================================================
# KONFIGURASI API (GABUNGAN)
# ==================================================
BASE_API = "http://localhost:8600"  # GANTI JIKA PORT BERBEDA

API_FB_PREDICT = f"{BASE_API}/predict_fb"
API_FB_TRAIN   = f"{BASE_API}/train_fb"

API_VM_PREDICT = f"{BASE_API}/predict_vm"
API_VM_TRAIN   = f"{BASE_API}/train_vm"

API_PC_PREDICT = f"{BASE_API}/predict_pc"
API_PC_TRAIN   = f"{BASE_API}/train_pc"
# ==================================================
# KONFIGURASI FILE DATA (UNTUK DROPDOWN DEFAULT)
# ==================================================
EXCEL_PATH = "05. Database RP May 2025 - AC REGISTER.xlsx"
SHEET_NAME = "Raw"

# =====================
# Konfigurasi
# =====================


RENAME_MAP_FB = {
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

SELECTED_FEATURES_FB = [
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

CATEGORICAL_COLS_FB = [
    "ROUNDTRIPROUTE",
    "AIRCRAFT_TYPE",
    "AC_REG",
    "SERVICE_TYPE",
    "FLIGHT_ROUTE",
]
NUMERIC_COLS_FB = [c for c in SELECTED_FEATURES_FB if c not in CATEGORICAL_COLS_FB]


# PC

RENAME_MAP_PC = {
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

SELECTED_FEATURES_PC = [
    "RPK_000_C_CLASS",
    "RTK_000",
    "RPK_000",
    "RTK_PASSENGER_000",
    "RPK_000_Y_CLASS",
    "ASK_000_C_CLASS",
    "PASSENGER_CARRIED_C_CLASS"
]

SELECTED_FEATURES_PC = SELECTED_FEATURES_PC + ["PASSENGER_COMMISSION"]

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Aviation Cost Prediction System",
    layout="wide",
)

# ==================================================
# SIDEBAR NAVIGATION (NESTED)
# ==================================================
st.sidebar.title("🧭 Navigation")

model_menu = st.sidebar.radio(
    "📦 Model",
    ["Fuel Burn", "Variable Maintenance", "Passenger Commission"],
)

action_menu = st.sidebar.radio(
    "⚙️ Action",
    ["Predict", "Train"],
)

st.sidebar.markdown("---")
st.sidebar.caption("Single API • XGBoost Models")

# ==================================================
# HELPER: LOAD DATA SAMPLE
# ==================================================

df_sample = pd.read_csv("sample/sample.csv")
df_filter = pd.read_csv("sample/df_filter.csv")

# ==================================================
# MAIN CONTENT
# ==================================================

# =========================
# FUEL BURN MODEL
# =========================
if model_menu == "Fuel Burn":

    st.title("✈️ Fuel Burn Prediction")

    if action_menu == "Predict":
        st.header("📈 Prediksi Fuel Burn (Liter)")

        df = df_sample.copy().rename(columns=RENAME_MAP_FB)[SELECTED_FEATURES_FB]
        df_filter_fb = df_filter.copy().rename(columns=RENAME_MAP_FB)
        sample_row = df.iloc[0]

        st.caption("Default value diisi dari salah satu contoh flight di dataset.")

        # =====================
        # Input fitur
        # =====================
        with st.form("predict_form"):
            col_cat1, col_cat2 = st.columns(2)
            col_cat3, col_cat4 = st.columns(2)
            col_cat5, _ = st.columns(2)

            # Opsi untuk categorical dari data
            rt_options = sorted(df_filter_fb["ROUNDTRIPROUTE"].dropna().unique())
            atype_options = sorted(df_filter_fb["AIRCRAFT_TYPE"].dropna().unique())
            ac_reg_options = sorted(df_filter_fb["AC_REG"].dropna().unique())
            stype_options = sorted(df_filter_fb["SERVICE_TYPE"].dropna().unique())
            froute_options = sorted(df_filter_fb["FLIGHT_ROUTE"].dropna().unique())

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
                        API_FB_PREDICT, json={"records": [record]}
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

    elif action_menu == "Train":
        st.header("🔁 Train / Retrain Model")

        st.write(
            "API akan membaca file Excel, melatih model XGBoost, "
            "menyimpan model ke disk, dan mengembalikan MAPE & RMSE."
        )

        st.warning("""
        ⚠️ Perhatian:
        - Proses training bisa memakan waktu (tergantung size dataset).
        - Model lama akan di-*overwrite* oleh model baru.
        """)

        if st.button("Train sekarang"):
            try:
                with st.spinner("Training model di server API..."):
                    resp = requests.post(API_FB_TRAIN)
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


# =========================
# VARIABLE MAINTENANCE MODEL
# =========================
elif model_menu == "Variable Maintenance":

    st.title("🛠️ Variable Maintenance Prediction")

    if action_menu == "Predict":
        st.header("🔮 Prediksi Variable Maintenance")
        st.write("Dropdown kategori otomatis menyesuaikan pilihan sebelumnya.")

        sample_row = df_sample.copy().fillna("").iloc[0]
        df_filter_vm = df_filter.copy().fillna("")

        # ============================
        # LANGKAH 1 — PILIH AC REG
        # ============================
        AC_REG = st.selectbox(
            "Pilih AC REG",
            sorted(df_filter_vm['AC REG'].astype(str).unique())
        )

        df_filter_vm = df_filter_vm[df_filter_vm['AC REG'] == AC_REG]

        # ============================
        # LANGKAH 2 — FILTER PERIODE
        # ============================
        PERIODE = st.selectbox(
            "Pilih PERIODE",
            sorted(df_filter_vm['PERIODE'].astype(str).unique())
        )

        df_filter_vm = df_filter_vm[df_filter_vm['PERIODE'] == PERIODE]

        # ============================
        # LANGKAH 3 — FILTER AIRCRAFT TYPE
        # ============================
        AIRCRAFT_TYPE = st.selectbox(
            "AIRCRAFT TYPE",
            sorted(df_filter_vm['AIRCRAFT TYPE'].astype(str).unique())
        )

        df_filter_vm = df_filter_vm[df_filter_vm['AIRCRAFT TYPE'] == AIRCRAFT_TYPE]

        # ============================
        # LANGKAH 4 — FILTER SERVICE TYPE
        # ============================
        SERVICE_TYPE = st.selectbox(
            "SERVICE TYPE",
            sorted(df_filter_vm['SERVICE TYPE'].astype(str).unique())
        )

        df_filter_vm = df_filter_vm[df_filter_vm['SERVICE TYPE'] == SERVICE_TYPE]

        # ============================
        # LANGKAH 5 — FILTER ROUNDTRIPROUTE
        # ============================
        ROUNDTRIPROUTE = st.selectbox(
            "ROUNDTRIPROUTE",
            sorted(df_filter_vm['ROUNDTRIPROUTE'].astype(str).unique())
        )

        df_filter_vm = df_filter_vm[df_filter_vm['ROUNDTRIPROUTE'] == ROUNDTRIPROUTE]

        # ============================
        # LANGKAH 6 — FILTER FLIGHT ROUTE
        # ============================
        FLIGHT_ROUTE = st.selectbox(
            "FLIGHT ROUTE",
            sorted(df_filter_vm['FLIGHT ROUTE'].astype(str).unique())
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
                    resp = requests.post(API_VM_PREDICT, json={"records": [record]})

                if resp.status_code == 200:
                    pred = resp.json()['predictions'][0]
                    st.success("Prediksi Berhasil 🎉")
                    st.metric("Prediksi Variable Maintenance ($)", f"{pred:,.2f}")
                    st.json(resp.json())
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")

            except Exception as e:
                st.error(f"Gagal menghubungi API: {e}")

    elif action_menu == "Train":
        st.header("🔁 Latih / Retrain Model Variable Maintenance")
        st.write("""
            Endpoint ini akan membaca dataset yang sudah ditentukan di API (`EXCEL_PATH`),
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
                    resp = requests.post(API_VM_TRAIN)

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



# =========================
# PASSENGER COMMISION MODEL
# =========================

elif model_menu == "Passenger Commission":

    st.title("✈️ Passenger Commission Predictor (LightGBM + API)")

    if action_menu == "Predict":
        st.header("📈 Prediksi Passenger Commission ($)")

        sample_row = df_sample.copy().rename(columns=RENAME_MAP_PC).iloc[0]
        
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
                        API_PC_PREDICT, json={"records": [record]}
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

    elif action_menu == "Train":
        st.header("🔁 Latih / Retrain Model Passenger Commision")
        st.write("""
            Endpoint ini akan membaca dataset yang sudah ditentukan di API (`EXCEL_PATH`),
            melakukan preprocessing, training ulang LightGBM, kemudian menyimpan model baru.
        """)

        st.warning("""
        ⚠️ Perhatian:
        - Proses training bisa memakan waktu (tergantung size dataset).
        - Model lama akan di-*overwrite* oleh model baru.
        """)

        if st.button("Train sekarang"):
            try:
                with st.spinner("Training model di server API..."):
                    resp = requests.post(API_PC_TRAIN)
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
