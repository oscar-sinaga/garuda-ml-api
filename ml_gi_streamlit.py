#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Streamlit App
- Fuel Burn Model (Predict & Train)
- Variable Maintenance Model (Predict & Train)
Menggunakan 1 FastAPI backend (API gabungan)
"""


# How to run:
# 1. uvicorn ml_gi_api:app --host 0.0.0.0 --port 8600 --reload
# 2. streamlit run ml_gi_streamlit.py


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

API_RESERVATION_PREDICT = f"{BASE_API}/predict_reservation"
API_RESERVATION_TRAIN   = f"{BASE_API}/train_reservation"
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


# RESERVATION

RENAME_MAP_RESERVATION = {
            'PASSENGER CARRIED': 'PASSENGER_CARRIED', 
            'PASSENGER CARRIED Y CLASS': 'PASSENGER_CARRIED_Y_CLASS', 
            'PASSENGER CARRIED C CLASS': 'PASSENGER_CARRIED_C_CLASS', 
            'CARGO CARRIED': 'CARGO_CARRIED',
            'FREIGHT CARRIED': 'FREIGHT_CARRIED',
            'PASSENGER COMMISSION': 'PASSENGER_COMMISSION',
            'BLOCK HOURS': 'BLOCK_HOURS',
            'RPK (000)': 'RPK_000', 
            'RPK (000) Y CLASS': 'RPK_000_Y_CLASS', 
            'SEAT OFFERED': 'SEAT_OFFERED', 
            'SEAT OFFERED Y CLASS': 'SEAT_OFFERED_Y_CLASS',
            'FLIGHT ROUTE': 'FLIGHT_ROUTE', 
            'SERVICE TYPE': 'SERVICE_TYPE', 
            'AIRCRAFT TYPE': 'AIRCRAFT_TYPE', 
            'Region': 'REGION'
        }

SELECTED_FEATURES_RESERVATION = ['PASSENGER_CARRIED', 
                     'PASSENGER_CARRIED_Y_CLASS', 
                     'PASSENGER_CARRIED_C_CLASS', 
                     'CARGO_CARRIED',
                     'RPK_000', 
                     'RPK_000_Y_CLASS', 
                     'SEAT_OFFERED', 
                     'SEAT_OFFERED_Y_CLASS',
                     'FLIGHT_ROUTE', 
                     'SERVICE_TYPE', 
                     'AIRCRAFT_TYPE', 
                     'REGION']

SELECTED_FEATURES_RESERVATION = SELECTED_FEATURES_RESERVATION + ["RESERVATION"]

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
    ["Fuel Burn", "Variable Maintenance", "Passenger Commission", "Reservation"],
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
df_filter1 = pd.read_csv("sample/df_filter1.csv")

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


# ============
# RESERVATION
# ============

elif model_menu == "Reservation":

    st.title("✈️ Reservation Predictor (XGBoost + API)")

    if action_menu == "Predict":
        st.header("📈 Prediksi Reservation ($)")
        
        sample_row = df_sample.copy().rename(columns=RENAME_MAP_RESERVATION).iloc[0]

        df_filter_res = df_filter1.copy().rename(columns=RENAME_MAP_RESERVATION)
        
        actual = sample_row["RESERVATION"]

        st.caption("Default value diisi dari salah satu contoh flight di dataset.")

        # =====================
        # Input fitur
        # =====================
        with st.form("predict_form"):
            col_cat1, col_cat2 = st.columns(2)
            col_cat3, col_cat4 = st.columns(2)
            col_cat5, _ = st.columns(2)

            stype_options = sorted(df_filter_res["SERVICE_TYPE"].dropna().unique())
            froute_options = sorted(df_filter_res["FLIGHT_ROUTE"].dropna().unique())
            actype_options = sorted(df_filter_res["AIRCRAFT_TYPE"].dropna().unique())
            region_options = sorted(df_filter_res["REGION"].dropna().unique())

            def default_index(options, value):
                try:
                    return list(options).index(value)
                except ValueError:
                    return 0
                
            SERVICE_TYPE = col_cat1.selectbox(
                "SERVICE_TYPE",
                stype_options,
                index=default_index(stype_options, sample_row["SERVICE_TYPE"])
            )

            AIRCRAFT_TYPE = col_cat2.selectbox(
                "AIRCRAFT_TYPE",
                actype_options,
                index=default_index(actype_options, sample_row["AIRCRAFT_TYPE"]),
            )

            FLIGHT_ROUTE = col_cat3.selectbox(
                "FLIGHT_ROUTE",
                froute_options,
                index=default_index(froute_options, sample_row["FLIGHT_ROUTE"]),
            )

            REGION = col_cat4.selectbox(
                "REGION",
                region_options,
                index=default_index(region_options, sample_row["REGION"]),
            )


            st.markdown("---")
            st.subheader("Fitur Numerik")

            num_cols1, num_cols2, num_cols3 = st.columns(3)

            PASSENGER_CARRIED = num_cols1.number_input(
                "PASSENGER_CARRIED", value=float(sample_row["PASSENGER_CARRIED"])
            )

            PASSENGER_CARRIED_Y_CLASS = num_cols2.number_input(
                "PASSENGER_CARRIED_Y_CLASS", value=float(sample_row["PASSENGER_CARRIED_Y_CLASS"])
            )

            PASSENGER_CARRIED_C_CLASS = num_cols3.number_input(
                "PASSENGER_CARRIED_C_CLASS", value=float(sample_row["PASSENGER_CARRIED_C_CLASS"])
            )

            CARGO_CARRIED =  num_cols1.number_input(
                "CARGO_CARRIED", value=float(sample_row["CARGO_CARRIED"])
            )
            
            RPK_000 = num_cols2.number_input(
                "RPK_000", value=float(sample_row["RPK_000"])
            )

            RPK_000_Y_CLASS =num_cols3.number_input(
                "RPK_000_Y_CLASS", value=float(sample_row["RPK_000_Y_CLASS"])
            ) 
            SEAT_OFFERED =num_cols1.number_input(
                "SEAT_OFFERED", value=float(sample_row["SEAT_OFFERED"])
            )

            SEAT_OFFERED_Y_CLASS =num_cols2.number_input(
                "SEAT_OFFERED_Y_CLASS", value=float(sample_row["SEAT_OFFERED_Y_CLASS"])
            )


            # =====================================
            # SUBMIT BUTTON
            # =====================================

            submitted = st.form_submit_button("🔮 Prediksi Reservation")

        if submitted:
            record = {
                'PASSENGER_CARRIED': PASSENGER_CARRIED, 
                'PASSENGER_CARRIED_Y_CLASS': PASSENGER_CARRIED_Y_CLASS, 
                'PASSENGER_CARRIED_C_CLASS': PASSENGER_CARRIED_C_CLASS, 
                'CARGO_CARRIED': CARGO_CARRIED,
                'RPK_000': RPK_000, 
                'RPK_000_Y_CLASS': RPK_000_Y_CLASS, 
                'SEAT_OFFERED': SEAT_OFFERED, 
                'SEAT_OFFERED_Y_CLASS': SEAT_OFFERED_Y_CLASS,
                'FLIGHT_ROUTE': FLIGHT_ROUTE, 
                'SERVICE_TYPE': SERVICE_TYPE, 
                'AIRCRAFT_TYPE': AIRCRAFT_TYPE,
                'REGION': REGION
            }

            try:
                with st.spinner("Meminta prediksi ke API..."):
                    resp = requests.post(
                        API_RESERVATION_PREDICT, json={"records": [record]}
                    )
                if resp.status_code == 200:
                    data = resp.json()
                    pred = data["predictions"][0]
                    st.success("Prediksi berhasil ✅")
                    st.metric("Perkiraan Reservation ($)", f"{pred:,.2f}")
                    st.json(data)
                    st.text( f"{actual:,.2f}")
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Gagal menghubungi API: {e}")

    elif action_menu == "Train":
        st.header("🔁 Latih / Retrain Model Reservation")
        st.write("""
            Endpoint ini akan membaca dataset yang sudah ditentukan di API (`EXCEL_PATH`),
            melakukan preprocessing, training ulang XGBoost, kemudian menyimpan model baru.
        """)

        st.warning("""
        ⚠️ Perhatian:
        - Proses training bisa memakan waktu (tergantung size dataset).
        - Model lama akan di-*overwrite* oleh model baru.
        """)

        if st.button("Train sekarang"):
            try:
                with st.spinner("Training model di server API..."):
                    resp = requests.post(API_RESERVATION_TRAIN)
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
# VARIABLE MAINTENANCE RESERVE
# =========================
elif model_menu == "Maintenance Reserve":

    st.title("🛡️ Maintenance Reserve Prediction")

    if action_menu == "Predict":
        st.header("🔮 Prediksi Maintenance Reserve")
        st.write("Masukkan parameter operasional untuk memprediksi cadangan maintenance.")


        sample_row = df_sample.copy().fillna("").iloc[0]
        df_filter_mr = df_filter.copy().fillna("")


        c_cat1, c_cat2, c_cat3 = st.columns(3)

        with c_cat1:
            AC_REG = st.selectbox(
                "Pilih AC REG",
                sorted(df_filter_mr['AC REG'].astype(str).unique())
            )
            df_filter_mr = df_filter_mr[df_filter_mr['AC REG'] == AC_REG]

        with c_cat2:
            PERIODE = st.selectbox(
                "Pilih PERIODE",
                sorted(df_filter_mr['PERIODE'].astype(str).unique())
            )
            df_filter_mr = df_filter_mr[df_filter_mr['PERIODE'] == PERIODE]

        with c_cat3:
            available_types = sorted(df_filter_mr['AIRCRAFT TYPE'].astype(str).unique())
            default_idx = 0
            
            AIRCRAFT_TYPE = st.selectbox(
                "AIRCRAFT TYPE",
                available_types if available_types else ["Unknown"]
            )

        st.markdown("---")


        st.markdown("### ✏️ Masukkan Fitur Operasional")
        
        def_fh = float(sample_row.get("FLIGHT HOURS", 0.0))
        def_landing = float(sample_row.get("NUMBER OF LANDING", 0.0))
        def_fuel = float(sample_row.get("FUEL BURN (IN LITER)", 0.0))
        def_atk = float(sample_row.get("ATK (000)", 0.0))
        def_lease = float(sample_row.get("LEASE AIRCRAFT", 0.0))

        col_num1, col_num2 = st.columns(2)

        with col_num1:
            FLIGHT_HOURS = st.number_input("FLIGHT HOURS", min_value=0.0, value=def_fh)
            NUMBER_OF_LANDING = st.number_input("NUMBER OF LANDING (Cycles)", min_value=0.0, value=def_landing)
            
            # Info tambahan untuk user (Calculated Feature)
            fh_per_cycle_display = 0.0
            if NUMBER_OF_LANDING > 0:
                fh_per_cycle_display = FLIGHT_HOURS / NUMBER_OF_LANDING
            st.caption(f"ℹ️ FH per Cycle (Auto): {fh_per_cycle_display:.4f}")

        with col_num2:
            FUEL_BURN_IN_LITER = st.number_input("FUEL BURN (IN LITER)", min_value=0.0, value=def_fuel)
            ATK_000 = st.number_input("ATK (000)", min_value=0.0, value=def_atk)
            LEASE_AIRCRAFT = st.number_input("LEASE AIRCRAFT (Rate/Value)", min_value=0.0, value=def_lease)


        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🔮 Prediksi MR Sekarang"):

            fh_per_cycle = 0.0
            if NUMBER_OF_LANDING != 0:
                fh_per_cycle = FLIGHT_HOURS / NUMBER_OF_LANDING

            record = {
                # Categorical
                "AC_REG": AC_REG,
                "PERIODE": PERIODE,
                "AIRCRAFT_TYPE": AIRCRAFT_TYPE,
                
                # Numeric
                "FLIGHT_HOURS": FLIGHT_HOURS,
                "FUEL_BURN_IN_LITER": FUEL_BURN_IN_LITER,
                "NUMBER_OF_LANDING": NUMBER_OF_LANDING,
                "ATK_000": ATK_000,
                "LEASE_AIRCRAFT": LEASE_AIRCRAFT,
                
                # Derived Feature (Penting karena masuk ke selected_features training)
                "FH_per_Cycle": fh_per_cycle
            }

            try:

                with st.spinner("Menghubungi API Maintenance Reserve..."):
                    # Pastikan variabel API_MR_PREDICT sudah didefinisikan di config
                    resp = requests.post(API_MR_PREDICT, json={"records": [record]})

                if resp.status_code == 200:
                    result = resp.json()
                    pred = result['predictions'][0]
                    
                    st.success("Prediksi Berhasil 🎉")
                    st.metric("Prediksi Maintenance Reserve ($)", f"{pred:,.2f}")
                    
                    with st.expander("🔍 Lihat Detail JSON Response"):
                        st.json(result)
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")

            except Exception as e:
                st.error(f"Gagal menghubungi API: {e}")

    elif action_menu == "Train":
        st.header("🔁 Latih / Retrain Model Maintenance Reserve")
        st.write("""
            Endpoint ini akan membaca dataset, melakukan grouping berdasarkan `AC REG` & `PERIODE`,
            menghitung `FH_per_Cycle`, dan melatih ulang model XGBoost untuk Maintenance Reserve.
        """)

        st.info(f"Fitur yang digunakan: FLIGHT HOURS, FUEL BURN, LANDING, ATK, LEASE, AIRCRAFT TYPE, AC REG, PERIODE, FH_per_Cycle.")

        if st.button("🚀 Train Model MR Sekarang"):
            try:
                with st.spinner("Sedang melatih model MR di server..."):
                    resp = requests.post(API_MR_TRAIN)

                if resp.status_code == 200:
                    st.success("Training Model MR Selesai! 🎉")

                    data = resp.json()

                    c_met1, c_met2, c_met3 = st.columns(3)
                    c_met1.metric("MAPE (%)", f"{data.get('mape_percent', 0):.2f}")
                    c_met2.metric("RMSE", f"{data.get('rmse', 0):.4f}")
                    c_met3.metric("R2 Score", f"{data.get('r2_score', 0):.4f}")

                    st.write(f"**Data Info:** Train Size: {data.get('n_train')}, Test Size: {data.get('n_test')}")
                    
                    st.subheader("📄 Detail Response")
                    st.json(data)

                else:
                    st.error(f"Training gagal — Error {resp.status_code}")
                    st.text(resp.text)

            except Exception as e:
                st.error(f"Gagal menghubungi API: {e}")
