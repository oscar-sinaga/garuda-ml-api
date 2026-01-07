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

API_FC_PREDICT = f"{BASE_API}/predict_fc"
API_FC_TRAIN = f"{BASE_API}/train_fc"

API_RESERVATION_PREDICT = f"{BASE_API}/predict_reservation"
API_RESERVATION_TRAIN   = f"{BASE_API}/train_reservation"

API_OBSC_PREDICT = f"{BASE_API}/predict_obsc"
API_OBSC_TRAIN = f"{BASE_API}/train_obsc"

API_MR_PREDICT = f"{BASE_API}/predict_mr"
API_MR_TRAIN   = f"{BASE_API}/train_mr"

API_CREW_PREDICT = f"{BASE_API}/predict_crew_fata"
API_CREW_TRAIN   = f"{BASE_API}/train_crew_fata"



API_BOFSC_TRAIN = f"{BASE_API}/train_bofsc"
API_BOFSC_PREDICT = f"{BASE_API}/predict_bofsc"


API_PAYROLL_PREDICT = f"{BASE_API}/predict_payroll"
API_PAYROLL_TRAIN   = f"{BASE_API}/train_payroll"

API_AFGH_PREDICT = f"{BASE_API}/predict_afgh"
API_AFGH_TRAIN   = f"{BASE_API}/train_afgh"

# ==================================================
# KONFIGURASI FILE DATA (UNTUK DROPDOWN DEFAULT)
# ==================================================
# EXCEL_PATH = "05. Database RP May 2025 - AC REGISTER.xlsx"
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

# FREIGHT COMMISSION

RENAME_MAP_FC = {
            'CLF (%)': 'CLF_PERCENT', 
            'CLF-GF (%)': 'CLF_GF_PERCENT', 
            'LOAD FACTOR (%)': 'LOAD_FACTOR_PERCENT', 
            'CARGO CARRIED': 'CARGO_CARRIED', 
            'FREIGHT CARRIED': 'FREIGHT_CARRIED',
            'SERVICE TYPE': 'SERVICE_TYPE', 
            'SUB-SERVICE': 'SUB_SERVICE', 
            'FLIGHT ROUTE': 'FLIGHT_ROUTE',
            'FREIGHT COMMISSION': 'FREIGHT_COMMISSION'
        }

SELECTED_FEATURES_FC = ['CLF_PERCENT', 
                        'CLF_GF_PERCENT', 
                        'LOAD_FACTOR_PERCENT', 
                        'CARGO_CARRIED', 
                        'FREIGHT_CARRIED',
                        'SERVICE_TYPE', 
                        'SUB_SERVICE', 
                        'FLIGHT_ROUTE'
]


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

# ON BOARD SERVICE AND CATERING
RENAME_MAP_OBSC = {
            # On Board Service
            'PASSENGER CARRIED': 'PASSENGER_CARRIED',
            'ATK PASSENGER (000)': 'ATK_PASSENGER_000', 
            'ASK (000)': 'ASK_000', 
            'ASK (000) Y CLASS': 'ASK_000_Y_CLASS', 
            'ATK (000)': 'ATK_000',
            'CABIN CREW PERSON': 'CABIN_CREW_PERSON', 
            'COCKPIT CREW PERSON': 'COCKPIT_CREW_PERSON',
            'BLOCK HOURS': 'BLOCK_HOURS',
            'ON BOARD SERVICE': 'ON_BOARD_SERVICE',
            'FLIGHT ROUTE': 'FLIGHT_ROUTE', 
            'SERVICE TYPE': 'SERVICE_TYPE', 
            'Region': 'REGION'
        }

SELECTED_FEATURES_OBS = ['PASSENGER_CARRIED',
                         'ATK_PASSENGER_000', 
                         'ASK_000', 
                         'ASK_000_Y_CLASS', 
                         'ATK_000',
                         'CABIN_CREW_PERSON', 
                         'COCKPIT_CREW_PERSON']

SELECTED_FEATURES_CATERING = ['PASSENGER_CARRIED', 
                            'ATK_000', 
                            'ATK_PASSENGER_000', 
                            'ASK_000', 
                            'ASK_000_Y_CLASS',
                            'FLIGHT_ROUTE', 
                            'SERVICE_TYPE', 
                            'REGION']

CATEGORICAL_COLS_CATERING = ['FLIGHT_ROUTE', 
                            'SERVICE_TYPE', 
                            'REGION']

# BRANCH OFFICE AND FIXED STATION COST
RENAME_MAP_BOFSC = {'SALES ORGANIZATION': 'SALES_ORGANIZATION', 
                    'COCKPIT CREW TRAVEL': 'COCKPIT_CREW_TRAVEL', 
                    'ASK (000) C CLASS': 'ASK_000_C_CLASS', 
                    'ASK (000)': 'ASK_000',
                    'CABIN CREW TRAVEL': 'CABIN_CREW_TRAVEL',
                    'Region': 'REGION', 
                    'GA Service': 'GA_SERVICE',
                    'ADMINISTRATION BO': 'ADMINISTRATION_BO',
                    'FLIGHT KILOMETERS': 'FLIGHT_KILOMETERS', 
                    'COCKPIT CREW PERSON': 'COCKPIT_CREW_PERSON',
                    }

SELECTED_FEATURES_ADMIN_BO = ['SALES_ORGANIZATION', 
                              'COCKPIT_CREW_TRAVEL', 
                              'ASK_000_C_CLASS', 
                              'ASK_000',
                              'CABIN_CREW_TRAVEL',
                              'PERIODE', 
                              'QUARTER', 
                              'REGION', 
                              'GA_SERVICE']

CATEGORICAL_COLS_ADMIN_BO = ['PERIODE', 
                              'QUARTER', 
                              'REGION', 
                              'GA_SERVICE']

SELECTED_FEATURES_STATION = ['COCKPIT_CREW_TRAVEL', 
                             'FLIGHT_KILOMETERS', 
                             'COCKPIT_CREW_PERSON', 
                             'CABIN_CREW_TRAVEL',
                             'PERIODE', 
                             'QUARTER', 
                             'REGION', 
                             'GA_SERVICE']

CATEGORICAL_COLS_STATION = ['PERIODE', 
                            'QUARTER', 
                            'REGION', 
                            'GA_SERVICE']
                    
# CREW FATA

RENAME_MAP_CREW = {
    'BLOCK HOURS': 'BLOCK_HOURS',          
    'FLIGHT KILOMETERS': 'FLIGHT_KILOMETERS',    
    'ASK (000)': 'ASK_000',            
    'NUMBER OF LANDING': 'NUMBER_OF_LANDING',    
    'AIRCRAFT TYPE': 'AIRCRAFT_TYPE',        
    'SERVICE TYPE': 'SERVICE_TYPE',         
    'PERIODE': 'PERIODE',
    'ATK (000)': 'ATK_000', 
    'SEAT OFFERED': 'SEAT_OFFERED',
    'COCKPIT CREW TRAVEL': 'COCKPIT_CREW_TRAVEL',
    'CABIN CREW TRAVEL': 'CABIN_CREW_TRAVEL'
}

SELECTED_FEATURES_CREW = [
    'BLOCK_HOURS', 'FLIGHT_KILOMETERS', 'ASK_000', 'NUMBER_OF_LANDING',
    'AIRCRAFT_TYPE', 'SERVICE_TYPE', 'PERIODE', 'ATK_000', 'SEAT_OFFERED'
]

# Payroll

RENAME_MAP_PAYROLL = {
    'BLOCK HOURS': 'BLOCK_HOURS', 'FLIGHT HOURS': 'FLIGHT_HOURS',
    'FLIGHT KILOMETERS': 'FLIGHT_KILOMETERS', 'NUMBER OF LANDING': 'NUMBER_OF_LANDING',
    'LEASE AIRCRAFT': 'LEASE_AIRCRAFT', 'AIRCRAFT TYPE': 'AIRCRAFT_TYPE',
    'SERVICE TYPE': 'SERVICE_TYPE', 'PERIODE': 'PERIODE', 'AC REG': 'AC_REG',
    'ASK (000) Y CLASS': 'ASK_000_Y_CLASS', 'ASK (000) C CLASS': 'ASK_000_C_CLASS',
    'FUEL BURN (IN LITER)': 'FUEL_BURN_IN_LITER',
    'COCKPIT CREW PERSON': 'COCKPIT_CREW_PERSON', 
    'CABIN CREW PERSON': 'CABIN_CREW_PERSON'
}


# Airport fees and ground handling
RENAME_MAP_AFGH = {
    'BLOCK HOURS': 'BLOCK_HOURS', 
    'ATK PASSENGER (000)': 'ATK_PASSENGER_000',
    'ATK (000)': 'ATK_000',
    'AIRCRAFT TYPE GROUPING': 'AIRCRAFT_TYPE_GROUPING',
    'FLIGHT ROUTE': 'FLIGHT_ROUTE',
    'AC REG': 'AC_REG',
    'AIRCRAFT TYPE': 'AIRCRAFT_TYPE'
}

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
    [
        "Fuel Burn", 
        "Variable Maintenance", 
        "Passenger Commission", 
        "Freight Commission",              
        "Reservation",
        "On Board Service and Catering",
        "Maintenance Reserve", 
        "Crew FATA",
        "Branch Office and Fixed Station Cost", 
        "Cabin & Crew Payroll",
        "Airport fees and ground handling"             
    ],
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

# ==================
# FREIGHT COMMISSION
# ==================

elif model_menu == "Freight Commission":

    st.title("✈️ Freight Commission Predictor (XGBoost + API)")

    if action_menu == "Predict":
        st.header("📈 Prediksi Freight Commission ($)")
        
        sample_row = df_sample.copy().rename(columns=RENAME_MAP_FC).iloc[0]

        df_filter_res = df_filter1.copy().rename(columns=RENAME_MAP_FC)
        
        actual = sample_row["FREIGHT_COMMISSION"]

        st.caption("Default value diisi dari salah satu contoh flight di dataset.")

        # =====================
        # Input fitur
        # =====================
        with st.form("predict_form"):
            col_cat1, col_cat2 = st.columns(2)
            col_cat3, col_cat4 = st.columns(2)

            stype_options = sorted(df_filter_res["SERVICE_TYPE"].dropna().unique())
            froute_options = sorted(df_filter_res["FLIGHT_ROUTE"].dropna().unique())
            subservice_options = sorted(df_filter_res["SUB_SERVICE"].dropna().unique())
            

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

            SUB_SERVICE = col_cat2.selectbox(
                "SUB_SERVICE",
                subservice_options,
                index=default_index(subservice_options, sample_row["SUB_SERVICE"])
            )

            FLIGHT_ROUTE = col_cat3.selectbox(
                "FLIGHT_ROUTE",
                froute_options,
                index=default_index(froute_options, sample_row["FLIGHT_ROUTE"])
            )

            st.markdown("---")
            st.subheader("Fitur Numerik")

            num_cols1, num_cols2, num_cols3 = st.columns(3)

            CLF_PERCENT = num_cols1.number_input(
                "CLF_PERCENT", value=float(sample_row["CLF_PERCENT"])
            )

            CLF_GF_PERCENT = num_cols2.number_input(
                "CLF_GF_PERCENT", value=float(sample_row["CLF_GF_PERCENT"])
            )

            LOAD_FACTOR_PERCENT = num_cols3.number_input(
                "LOAD_FACTOR_PERCENT", value=float(sample_row["LOAD_FACTOR_PERCENT"])
            )

            CARGO_CARRIED = num_cols1.number_input(
                "CARGO_CARRIED", value=float(sample_row["CARGO_CARRIED"])
            )

            FREIGHT_CARRIED = num_cols2.number_input(
                "FREIGHT_CARRIED", value=float(sample_row["FREIGHT_CARRIED"])
            )

            submitted = st.form_submit_button("🔮 Prediksi Freight Commission")

        if submitted:
            record = {
                "CLF_PERCENT": CLF_PERCENT,
                "CLF_GF_PERCENT": CLF_GF_PERCENT,
                "LOAD_FACTOR_PERCENT": LOAD_FACTOR_PERCENT,
                "CARGO_CARRIED": CARGO_CARRIED,
                "FREIGHT_CARRIED": FREIGHT_CARRIED,
                "SERVICE_TYPE": SERVICE_TYPE,
                "SUB_SERVICE": SUB_SERVICE,
                "FLIGHT_ROUTE": FLIGHT_ROUTE
            }

            try:
                with st.spinner("Meminta prediksi ke API..."):
                    resp = requests.post(
                        API_FC_PREDICT, json={"records": [record]}
                    )
                if resp.status_code == 200:
                    data = resp.json()
                    pred = data["predictions"][0]
                    st.success("Prediksi berhasil ✅")
                    st.metric("Perkiraan Freight Commission", f"{pred:,.2f}")
                    st.json(data)
                    st.text( f"{actual:,.2f}")
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Gagal menghubungi API: {e}")

    elif action_menu == "Train":
        st.header("🔁 Latih / Retrain Model Freight Commision")
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
                    resp = requests.post(API_FC_TRAIN)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success("Training selesai ✅")
                    st.json(data)
                    st.metric("MAPE (%)", f"{data['mape_percent']:.2f}")
                    st.metric("RMSE", f"{data['rmse']:.2f}")
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
                    st.metric("RMSE", f"{data['rmse']:.2f}")
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
        
        if st.button("🔮 Prediksi Maintenanc Reserve"):

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
                
                "FH_per_Cycle": fh_per_cycle
            }

            try:

                with st.spinner("Menghubungi API Maintenance Reserve..."):
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
        st.write("""
            Endpoint ini akan membaca dataset yang sudah ditentukan di API (`EXCEL_PATH`),
            melakukan preprocessing, training ulang XGBoost, kemudian menyimpan model baru.
        """)

        st.warning("""
        ⚠️ Perhatian:
        - Proses training bisa memakan waktu (tergantung size dataset).
        - Model lama akan di-*overwrite* oleh model baru.
        """)

        if st.button("Train Sekarang"):
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

# =============================
# ON BOARD SERVICE AND CATERING
# =============================

elif model_menu == "On Board Service and Catering":

    st.title("🍕 On Board Service and Catering Prediction")

    if action_menu == "Predict":
        st.subheader("🔮 Prediksi On Board Service dan Catering ($)")
        st.write("Masukkan parameter operasional untuk memprediksi On Board Service and Catering.")

        sample_row = df_sample.copy().fillna("").rename(columns=RENAME_MAP_OBSC).iloc[0]
        df_filter_obsc = df_filter1.copy().fillna("").rename(columns=RENAME_MAP_OBSC)

        # c_cat1, c_cat2, c_cat3 = st.columns(3)


        actual_obs = sample_row["ON_BOARD_SERVICE"]
        actual_catering = sample_row["CATERING"]

        st.caption("Default value diisi dari salah satu contoh flight di dataset.")

        # =====================
        # Input fitur
        # =====================
        with st.form("predict_form"):
            st.subheader("On Board Service")

            num_cols1, num_cols2, num_cols3 = st.columns(3)

            PASSENGER_CARRIED = num_cols1.number_input(
                "PASSENGER_CARRIED", value=int(sample_row["PASSENGER_CARRIED"])
            )

            ATK_000 = num_cols2.number_input(
                "ATK_000", value=float(sample_row["ATK_000"])
            )

            ATK_PASSENGER_000 = num_cols3.number_input(
                "ATK_PASSENGER_000", value=float(sample_row["ATK_PASSENGER_000"])
            )

            ASK_000 = num_cols1.number_input(
                "ASK_000", value=float(sample_row["ASK_000"])
            )

            ASK_000_Y_CLASS = num_cols2.number_input(
                "ASK_000_Y_CLASS", value=float(sample_row["ASK_000_Y_CLASS"])
            )

            CABIN_CREW_PERSON = num_cols3.number_input(
                "CABIN_CREW_PERSON", value=float(sample_row["CABIN_CREW_PERSON"])
            )

            COCKPIT_CREW_PERSON = num_cols1.number_input(
                "COCKPIT_CREW_PERSON", value=float(sample_row["COCKPIT_CREW_PERSON"])
            )

            st.markdown("---")
            st.subheader("Catering")

            num_cols1, num_cols2, num_cols3 = st.columns(3)

            froute_options = sorted(df_filter_obsc["FLIGHT_ROUTE"].dropna().unique())
            stype_options = sorted(df_filter_obsc["SERVICE_TYPE"].dropna().unique())
            region_options = sorted(df_filter_obsc["REGION"].dropna().unique())

            def default_index(options, value):
                try:
                    return list(options).index(value)
                except ValueError:
                    return 0
                
            FLIGHT_ROUTE = num_cols1.selectbox(
                "FLIGHT_ROUTE",
                froute_options,
                index=default_index(froute_options, sample_row["FLIGHT_ROUTE"]),
            )

            REGION = num_cols2.selectbox(
                "REGION",
                region_options,
                index=default_index(region_options, sample_row["REGION"]),
            )

            SERVICE_TYPE = num_cols3.selectbox(
                "SERVICE_TYPE",
                stype_options,
                index=default_index(stype_options, sample_row["SERVICE_TYPE"]),
            )

            submitted = st.form_submit_button("🔮 Prediksi OBSC")

        if submitted:
            record_obs = {
                'PASSENGER_CARRIED': PASSENGER_CARRIED,
                'ATK_PASSENGER_000': ATK_PASSENGER_000,
                'ASK_000': ASK_000,
                'ASK_000_Y_CLASS': ASK_000_Y_CLASS,
                'ATK_000': ATK_000,
                'CABIN_CREW_PERSON': CABIN_CREW_PERSON,
                'COCKPIT_CREW_PERSON': COCKPIT_CREW_PERSON
                }

            record_catering = {
                'PASSENGER_CARRIED': PASSENGER_CARRIED,
                'ATK_000': ATK_000,
                'ATK_PASSENGER_000': ATK_PASSENGER_000,
                'ASK_000': ASK_000,
                'ASK_000_Y_CLASS': ASK_000_Y_CLASS,
                'FLIGHT_ROUTE': FLIGHT_ROUTE,
                'SERVICE_TYPE': SERVICE_TYPE,
                'REGION': REGION
            } 

            try:
                with st.spinner("Meminta prediksi ke API..."):
                    resp = requests.post(
                        API_OBSC_PREDICT, 
                        json={
                                "obs": {"records": [record_obs]},
                                "catering": {"records": [record_catering]},
                            }
                    )
                if resp.status_code == 200:
                    data = resp.json()
                    pred_obs = data["obs"]["predictions"][0]
                    pred_catering = data["catering"]["predictions"][0]
                    st.success("Prediksi berhasil ✅")
                    st.metric("Perkiraan On Board Service ($)", f"{pred_obs:,.2f}")
                    st.metric("Perkiraan Catering ($)", f"{pred_catering:,.2f}")
                    st.json(data)
                    st.text( f"{actual_obs:,.2f}")
                    st.text( f"{actual_catering:,.2f}")
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Gagal menghubungi API: {e}")

    if action_menu == "Train":
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
                    resp = requests.post(API_OBSC_TRAIN)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success("Training selesai ✅")
                    st.json(data)
                    st.metric("MAPE On Board Service (%)", f"{data['obs']['mape_percent']:.2f}")
                    st.metric("RMSE On Board Service", f"{data['obs']['rmse']:.2f}")

                    st.metric("MAPE Catering (%)", f"{data['catering']['mape_percent']:.2f}")
                    st.metric("RMSE Catering", f"{data['catering']['rmse']:.2f}")
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Gagal menghubungi API: {e}")

# =========================
# CREW FATA
# =========================

elif model_menu == "Crew FATA":

    st.title("Crew FATA Prediction")

    if action_menu == "Predict":
        st.header("👨‍✈️ Prediksi Biaya Travel Crew")
        st.write("Masukkan parameter untuk memprediksi biaya **Cockpit Crew** dan **Cabin Crew**.")

        # Load sample data untuk default value
        sample_row = df_sample.copy().fillna("").rename(columns=RENAME_MAP_CREW).iloc[0]
        df_filter_crew = df_filter.copy().fillna("").rename(columns=RENAME_MAP_CREW)

        actual_cockpit = sample_row.get("COCKPIT_CREW_TRAVEL", 0)
        actual_cabin = sample_row.get("CABIN_CREW_TRAVEL", 0)

        with st.form("predict_crew_form"):
            st.subheader("Parameter Operasional")
            
            # Categorical Inputs
            c1, c2, c3 = st.columns(3)
            
            # Helper options
            ac_types = sorted(df_filter_crew["AIRCRAFT_TYPE"].astype(str).unique())
            srv_types = sorted(df_filter_crew["SERVICE_TYPE"].astype(str).unique())
            periods = sorted(df_filter_crew["PERIODE"].astype(str).unique())

            def get_idx(opts, val):
                try: return list(opts).index(str(val))
                except: return 0

            AIRCRAFT_TYPE = c1.selectbox("AIRCRAFT TYPE", ac_types, index=get_idx(ac_types, sample_row["AIRCRAFT_TYPE"]))
            SERVICE_TYPE = c2.selectbox("SERVICE TYPE", srv_types, index=get_idx(srv_types, sample_row["SERVICE_TYPE"]))
            PERIODE = c3.selectbox("PERIODE", periods, index=get_idx(periods, sample_row["PERIODE"]))

            st.markdown("---")
            st.subheader("Fitur Numerik")

            n1, n2, n3 = st.columns(3)

            BLOCK_HOURS = n1.number_input("BLOCK HOURS", min_value=0.0, value=float(sample_row["BLOCK_HOURS"]))
            FLIGHT_KILOMETERS = n2.number_input("FLIGHT KILOMETERS", min_value=0.0, value=float(sample_row["FLIGHT_KILOMETERS"]))
            ASK_000 = n3.number_input("ASK (000)", min_value=0.0, value=float(sample_row["ASK_000"]))

            NUMBER_OF_LANDING = n1.number_input("NUMBER OF LANDING", min_value=0.0, value=float(sample_row["NUMBER_OF_LANDING"]))
            ATK_000 = n2.number_input("ATK (000)", min_value=0.0, value=float(sample_row["ATK_000"]))
            SEAT_OFFERED = n3.number_input("SEAT OFFERED", min_value=0.0, value=float(sample_row["SEAT_OFFERED"]))

            submitted = st.form_submit_button("🚀 Prediksi Sekarang")

        if submitted:
            record = {
                "BLOCK_HOURS": BLOCK_HOURS,
                "FLIGHT_KILOMETERS": FLIGHT_KILOMETERS,
                "ASK_000": ASK_000,
                "NUMBER_OF_LANDING": NUMBER_OF_LANDING,
                "ATK_000": ATK_000,
                "SEAT_OFFERED": SEAT_OFFERED,
                "AIRCRAFT_TYPE": AIRCRAFT_TYPE,
                "SERVICE_TYPE": SERVICE_TYPE,
                "PERIODE": PERIODE
            }

            try:
                with st.spinner("Menghubungi API Crew FATA..."):
                    resp = requests.post(API_CREW_PREDICT, json={"records": [record]})

                if resp.status_code == 200:
                    res = resp.json()
                    pred_cockpit = res['cockpit_predictions'][0]
                    pred_cabin = res['cabin_predictions'][0]

                    st.success("Prediksi Selesai! ✅")
                    
                    col_res1, col_res2 = st.columns(2)

                    with col_res1:
                        st.info("👨‍✈️ **COCKPIT CREW TRAVEL**")
                        st.metric("Prediksi ($)", f"{pred_cockpit:,.2f}")
                        st.caption(f"Actual (Sample): {actual_cockpit:,.2f}")

                    with col_res2:
                        st.info("👩‍✈️ **CABIN CREW TRAVEL**")
                        st.metric("Prediksi ($)", f"{pred_cabin:,.2f}")
                        st.caption(f"Actual (Sample): {actual_cabin:,.2f}")
                    
                    st.json(res)

                else:
                    st.error(f"API Error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")


    elif action_menu == "Train":
        
        st.write("Endpoint ini akan membaca dataset yang sudah ditentukan di API (EXCEL_PATH), melakukan preprocessing, training ulang XGBoost, kemudian menyimpan model baru.")

        st.warning("""
        ⚠️ Perhatian:
        - Proses training bisa memakan waktu (tergantung size dataset).
        - Model lama akan di-*overwrite* oleh model baru.
        """)

        if st.button("Train Sekarang"):
            try:
                with st.spinner("Sedang melatih model Cockpit & Cabin... Mohon tunggu..."):
                    resp = requests.post(API_CREW_TRAIN)
                
                if resp.status_code == 200:
                    data = resp.json()
                    st.success("Training Selesai! ✅")

                    # Tampilkan hasil side-by-side
                    c1, c2 = st.columns(2)

                    with c1:
                        st.subheader("👨‍✈️ Cockpit Model")
                        st.metric("MAPE (%)", f"{data['cockpit']['mape_percent']:.2f}%")
                        st.metric("RMSE", f"{data['cockpit']['rmse']:.2f}")
                        st.write(f"Train/Test: {data['cockpit']['n_train']} / {data['cockpit']['n_test']}")

                    with c2:
                        st.subheader("👩‍✈️ Cabin Model")
                        st.metric("MAPE (%)", f"{data['cabin']['mape_percent']:.2f}%")
                        st.metric("RMSE", f"{data['cabin']['rmse']:.2f}")
                        st.write(f"Train/Test: {data['cabin']['n_train']} / {data['cabin']['n_test']}")

                    with st.expander("Lihat Full Response JSON"):
                        st.json(data)
                
                else:
                    st.error(f"Training Gagal: {resp.text}")

            except Exception as e:
                st.error(f"Gagal menghubungi API: {e}")

# ====================================
# BRANCH OFFICE AND FIXED STATION COST
# (ADMINISTRATION BO and STATION)
# ====================================

# Administration BO
elif model_menu == "Branch Office and Fixed Station Cost":
    st.title("✈️ Branch Office and Fixed Station Cost Predictor (XGBoost + API)")

    if action_menu == "Predict":
        st.header("📈 Prediksi Branch Office and Fixed Station Cost ($)")
        
        sample_row = df_sample.copy().rename(columns=RENAME_MAP_BOFSC).iloc[0]

        df_filter_bofsc = df_filter1.copy().rename(columns=RENAME_MAP_BOFSC)
        
        actual_admin_bo = sample_row["ADMINISTRATION_BO"]
        actual_station = sample_row["STATION"]
        
        st.caption("Default value diisi dari salah satu contoh flight di dataset.")

        # =====================
        # Input fitur
        # =====================

        with st.form("predict_form"):
            st.subheader("Administration BO")

            st.markdown("#### Categorical Features")
            num_cols1, num_cols2, num_cols3 = st.columns(3)

            # Categorical
            periode_options = sorted(df_filter_bofsc["PERIODE"].dropna().unique())
            quarter_options = sorted(df_filter_bofsc["QUARTER"].dropna().unique())
            region_options = sorted(df_filter_bofsc["REGION"].dropna().unique())
            ga_service_options = sorted(df_filter_bofsc["GA_SERVICE"].dropna().unique())

            def default_index(options, value):
                try:
                    return list(options).index(value)
                except ValueError:
                    return 0
                
            PERIODE = num_cols1.selectbox(
                "PERIODE",
                periode_options,
                index=default_index(periode_options, sample_row["PERIODE"]),
            )

            QUARTER = num_cols2.selectbox(
                "QUARTER",
                periode_options,
                index=default_index(quarter_options, sample_row["QUARTER"]),
            )

            REGION = num_cols3.selectbox(
                "REGION",
                region_options,
                index=default_index(periode_options, sample_row["REGION"]),
            )

            GA_SERVICE = num_cols1.selectbox(
                "GA_SERVICE",
                ga_service_options,
                index=default_index(periode_options, sample_row["GA_SERVICE"]),
            )

            # Numeric

            st.markdown("#### Numerical Features")
            num_cols1, num_cols2, num_cols3 = st.columns(3)

            SALES_ORGANIZATION = num_cols1.number_input(
                "SALES_ORGANIZATION", value=float(sample_row["SALES_ORGANIZATION"])
            )

            COCKPIT_CREW_TRAVEL = num_cols2.number_input(
                "COCKPIT_CREW_TRAVEL", value=float(sample_row["COCKPIT_CREW_TRAVEL"])
            )

            ASK_000_C_CLASS = num_cols3.number_input(
                "ASK_000_C_CLASS", value=float(sample_row["ASK_000_C_CLASS"])
            )
            
            ASK_000 = num_cols1.number_input(
                "ASK_000", value=float(sample_row["ASK_000"])
            )

            CABIN_CREW_TRAVEL = num_cols2.number_input(
                "CABIN_CREW_TRAVEL", value=float(sample_row["CABIN_CREW_TRAVEL"])
            )

            st.markdown("---")
            st.subheader("Station")

            FLIGHT_KILOMETERS = num_cols1.number_input(
                "FLIGHT_KILOMETERS", value=float(sample_row["FLIGHT_KILOMETERS"])
            )
            
            COCKPIT_CREW_PERSON = num_cols2.number_input(
                "COCKPIT_CREW_PERSON", value=float(sample_row["COCKPIT_CREW_PERSON"])
            )

            submitted = st.form_submit_button("🔮 Prediksi BOFSC")

        if submitted:
            record_admin_bo = {
                'SALES_ORGANIZATION': SALES_ORGANIZATION,
                'COCKPIT_CREW_TRAVEL': COCKPIT_CREW_TRAVEL,
                'ASK_000': ASK_000,
                'ASK_000_C_CLASS': ASK_000_C_CLASS,
                'CABIN_CREW_TRAVEL': CABIN_CREW_TRAVEL,
                'PERIODE': PERIODE,
                'QUARTER': QUARTER,
                'REGION': REGION,
                'GA_SERVICE': GA_SERVICE
                }
            
            record_station = {
                'COCKPIT_CREW_TRAVEL': COCKPIT_CREW_TRAVEL,
                'FLIGHT_KILOMETERS': FLIGHT_KILOMETERS,
                'COCKPIT_CREW_PERSON': COCKPIT_CREW_PERSON,
                'CABIN_CREW_TRAVEL': CABIN_CREW_TRAVEL,
                'PERIODE': PERIODE,
                'QUARTER': QUARTER,
                'REGION': REGION,
                'GA_SERVICE': GA_SERVICE
            }
            
            try:
                with st.spinner("Meminta prediksi ke API..."):
                    resp = requests.post(
                        API_BOFSC_PREDICT, 
                        json={
                                "administration_bo": {"records": [record_admin_bo]},
                                "station": {"records": [record_station]},
                            }
                    )
                if resp.status_code == 200:
                    data = resp.json()
                    pred_admin_bo = data["administration_bo"]["predictions"][0]
                    pred_station= data["station"]["predictions"][0]
                    st.success("Prediksi berhasil ✅")
                    st.metric("Perkiraan Administration BO ($)", f"{pred_admin_bo:,.2f}")
                    st.metric("Perkiraan Fixed Station Cost ($)", f"{pred_station:,.2f}")
                    st.json(data)
                    st.text( f"{actual_admin_bo:,.2f}")
                    st.text( f"{actual_station:,.2f}")
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Gagal menghubungi API: {e}")

    elif action_menu == "Train":
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
                    resp = requests.post(API_BOFSC_TRAIN)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success("Training selesai ✅")
                    st.json(data)
                    st.metric("MAPE Administration BO (%)", f"{data['administration_bo']['mape_percent']:.2f}")
                    st.metric("RMSE Administration BO", f"{data['administration_bo']['rmse']:.2f}")

                    st.metric("MAPE Station (%)", f"{data['station']['mape_percent']:.2f}")
                    st.metric("RMSE Station", f"{data['station']['rmse']:.2f}")
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Gagal menghubungi API: {e}")

#=============================================
# PAYROLL
#=============================================


elif model_menu == "Cabin & Crew Payroll":
    st.title("💸 Cabin & Crew Payroll Prediction")

    if action_menu == "Predict":
        st.header("🔮 Prediksi Payroll Cost")
        st.markdown("Memprediksi **Cockpit Crew Person** & **Cabin Crew Person** Cost menggunakan XGBoost")

        
        # Sample data setup
        sample_row = df_sample.copy().fillna(0).rename(columns=RENAME_MAP_PAYROLL).iloc[0]
        df_filter_pay = df_filter.copy().fillna("").rename(columns=RENAME_MAP_PAYROLL)
        
        act_cp = sample_row.get("COCKPIT_CREW_PERSON", 0)
        act_cb = sample_row.get("CABIN_CREW_PERSON", 0)
        
        with st.form("form_payroll"):
            st.subheader("Categorical")
            c1, c2, c3 = st.columns(3)
            
            # Helper options
            ac_types = sorted(df_filter_pay["AIRCRAFT_TYPE"].astype(str).unique())
            srv_types = sorted(df_filter_pay["SERVICE_TYPE"].astype(str).unique())
            periods = sorted(df_filter_pay["PERIODE"].astype(str).unique())
            ac_regs = sorted(df_filter_pay["AC_REG"].astype(str).unique()) if "AC_REG" in df_filter_pay.columns else ["PK-GAA"]
            
            def idx(opts, val):
                try: return list(opts).index(str(val))
                except: return 0

            AIRCRAFT_TYPE = c1.selectbox("AIRCRAFT TYPE", ac_types, index=idx(ac_types, sample_row["AIRCRAFT_TYPE"]))
            SERVICE_TYPE = c2.selectbox("SERVICE TYPE", srv_types, index=idx(srv_types, sample_row["SERVICE_TYPE"]))
            PERIODE = c3.selectbox("PERIODE", periods, index=idx(periods, sample_row["PERIODE"]))
            AC_REG = st.selectbox("AC REG (Optional Identity)", ac_regs, index=0)
            
            st.markdown("---")
            st.subheader("Numerical")
            
            # Input Numerik Gabungan
            n1, n2, n3 = st.columns(3)
            BLOCK_HOURS = n1.number_input("BLOCK HOURS", min_value=0.0, value=float(sample_row["BLOCK_HOURS"]))
            FLIGHT_HOURS = n2.number_input("FLIGHT HOURS", min_value=0.0, value=float(sample_row["FLIGHT_HOURS"]))
            FLIGHT_KILOMETERS = n3.number_input("FLIGHT KILOMETERS", min_value=0.0, value=float(sample_row["FLIGHT_KILOMETERS"]))
            
            n4, n5, n6 = st.columns(3)
            NUMBER_OF_LANDING = n4.number_input("NUMBER OF LANDING", min_value=0.0, value=float(sample_row["NUMBER_OF_LANDING"]))
            LEASE_AIRCRAFT = n5.number_input("LEASE AIRCRAFT", min_value=0.0, value=float(sample_row["LEASE_AIRCRAFT"]))
            FUEL_BURN_IN_LITER = n6.number_input("FUEL BURN (LITER)", min_value=0.0, value=float(sample_row["FUEL_BURN_IN_LITER"]))
            
            n7, n8 = st.columns(2)
            ASK_000_Y_CLASS = n7.number_input("ASK (000) Y CLASS", min_value=0.0, value=float(sample_row["ASK_000_Y_CLASS"]))
            ASK_000_C_CLASS = n8.number_input("ASK (000) C CLASS", min_value=0.0, value=float(sample_row["ASK_000_C_CLASS"]))
            
            submit = st.form_submit_button("Train Sekarang")
            
        if submit:
            record = {
                "AC_REG": AC_REG, "PERIODE": PERIODE,
                "AIRCRAFT_TYPE": AIRCRAFT_TYPE, "SERVICE_TYPE": SERVICE_TYPE,
                "BLOCK_HOURS": BLOCK_HOURS, "FLIGHT_HOURS": FLIGHT_HOURS,
                "FLIGHT_KILOMETERS": FLIGHT_KILOMETERS, "NUMBER_OF_LANDING": NUMBER_OF_LANDING,
                "LEASE_AIRCRAFT": LEASE_AIRCRAFT, "FUEL_BURN_IN_LITER": FUEL_BURN_IN_LITER,
                "ASK_000_Y_CLASS": ASK_000_Y_CLASS, "ASK_000_C_CLASS": ASK_000_C_CLASS
            }
            
            try:
                with st.spinner("Meminta prediksi ke API..."):
                    resp = requests.post(API_PAYROLL_PREDICT, json={"records": [record]})
                    
                if resp.status_code == 200:
                    res = resp.json()
                    val_cp = res["cockpit_person_cost"][0]
                    val_cb = res["cabin_person_cost"][0]
                    
                    st.success("Prediksi Selesai! ✅")
                    
                    c_res1, c_res2 = st.columns(2)
                    with c_res1:
                        st.info("👨‍✈️ **COCKPIT CREW PERSON**")
                        st.metric("Prediction ($)", f"{val_cp:,.2f}")
                        st.caption(f"Actual Sample: {act_cp:,.2f}")
                        
                    with c_res2:
                        st.info("👩‍✈️ **CABIN CREW PERSON**")
                        st.metric("Prediction ($)", f"{val_cb:,.2f}")
                        st.caption(f"Actual Sample: {act_cb:,.2f}")
                        
                    st.json(res)
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")

    elif action_menu == "Train":
        st.write("""
            Endpoint ini akan membaca dataset yang sudah ditentukan di API (`EXCEL_PATH`),
            melakukan preprocessing, training ulang XGBoost, kemudian menyimpan model baru.
        """)

        st.warning("""
        ⚠️ Perhatian:
        - Proses training bisa memakan waktu (tergantung size dataset).
        - Model lama akan di-*overwrite* oleh model baru.
        """)
        
        if st.button("Train Sekarang"):
            try:
                with st.spinner("Training model di server API.."):
                    resp = requests.post(API_PAYROLL_TRAIN)
                    
                if resp.status_code == 200:
                    data = resp.json()
                    st.success("Training Berhasil! ✅")
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.subheader("👨‍✈️ Cockpit Model")
                        st.metric("MAPE (%)", f"{data['cockpit']['mape_percent']:.2f}%")
                        st.metric("RMSE", f"{data['cockpit']['rmse']:.2f}")
                    
                    with c2:
                        st.subheader("👩‍✈️ Cabin Model")
                        st.metric("MAPE (%)", f"{data['cabin']['mape_percent']:.2f}%")
                        st.metric("RMSE", f"{data['cabin']['rmse']:.2f}")
                        
                    with st.expander("Detail JSON Response"):
                        st.json(data)
                else:
                    st.error(f"Training Failed: {resp.text}")
            except Exception as e:
                st.error(f"Error: {e}")



# ====================================
# Airport fees and ground handling COST
# (LANDING, HANDLING, AIRCRAFT TRAFFIC CONTROL)
# ====================================

# Administration BO
elif model_menu == "Airport fees and ground handling":
    st.title("✈️ Airport fees and ground handling Predictor (XGBoost + API)")

    if action_menu == "Predict":
        st.header("📈 Prediksi Airport fees and ground handling (LANDING, HANDLING, AIR TRAFFIC CONTROL) ($)")
        
        sample_row = df_sample.copy().rename(columns=RENAME_MAP_AFGH).iloc[0]

        df_filter_afgh = df_filter1.copy().rename(columns=RENAME_MAP_AFGH)
        
        # actual_landing = sample_row["LANDING"]
        # actual_handling = sample_row["HANDLING"]
        # actual_atc = sample_row["AIRCRAFT_TRAFFIC_CONTROL"]
        
        st.caption("Default value diisi dari salah satu contoh flight di dataset.")

        # =====================
        # Input fitur
        # =====================

        with st.form("predict_form"):
            st.markdown("#### Categorical Features")
            num_cols1, num_cols2, num_cols3 = st.columns(3)

            # Categorical
            aircraft_type_grouping_options = sorted(df_filter_afgh["AIRCRAFT_TYPE_GROUPING"].dropna().unique())
            flight_route_options = sorted(df_filter_afgh["FLIGHT_ROUTE"].dropna().unique())
            roundtriproute_options = sorted(df_filter_afgh["ROUNDTRIPROUTE"].dropna().unique())
            ac_reg_options = sorted(df_filter_afgh["AC_REG"].dropna().unique())
            aircraft_type_options = sorted(df_filter_afgh["AIRCRAFT_TYPE"].dropna().unique())
            
            def default_index(options, value):
                try:
                    return list(options).index(value)
                except ValueError:
                    return 0

            AIRCRAFT_TYPE_GROUPING = num_cols2.selectbox(
                "AIRCRAFT_TYPE_GROUPING",
                aircraft_type_grouping_options,
                index=default_index(aircraft_type_grouping_options, sample_row["AIRCRAFT_TYPE_GROUPING"]),
            )
            ROUNDTRIPROUTE = num_cols3.selectbox(
                "ROUNDTRIPROUTE",
                roundtriproute_options,
                index=default_index(roundtriproute_options, sample_row["ROUNDTRIPROUTE"]),
            )
            FLIGHT_ROUTE = num_cols3.selectbox(
                "FLIGHT_ROUTE",
                flight_route_options,
                index=default_index(flight_route_options, sample_row["FLIGHT_ROUTE"]),
            )

            AC_REG = num_cols1.selectbox(
                "AC_REG",
                ac_reg_options,
                index=default_index(ac_reg_options, sample_row["AC_REG"]),
            )

            AIRCRAFT_TYPE = num_cols2.selectbox(
                "AIRCRAFT_TYPE",
                aircraft_type_options,
                index=default_index(aircraft_type_options, sample_row["AIRCRAFT_TYPE"]),
            )

        
            # Numeric
            st.markdown("---")
            st.markdown("#### Numerical Features")
            num_cols1, num_cols2, num_cols3 = st.columns(3)

            ATK_PASSENGER_000 = num_cols3.number_input(
                "ATK_PASSENGER_000", value=float(sample_row["ATK_PASSENGER_000"])
            )

            ATK_000 = num_cols1.number_input(
                "ATK_000", value=float(sample_row["ATK_000"])
            )

            BLOCK_HOURS = num_cols2.number_input(
                "BLOCK_HOURS", value=float(sample_row["BLOCK_HOURS"])
            )



            submitted = st.form_submit_button("🔮 Prediksi AIRPORT FEES AND GROUND HANDLING")

        if submitted:
            record_landing = {
                'ATK_PASSENGER_000': ATK_PASSENGER_000,
                'ATK_000': ATK_000,
                'BLOCK_HOURS': BLOCK_HOURS,
                'AIRCRAFT_TYPE_GROUPING': AIRCRAFT_TYPE_GROUPING,
                'FLIGHT_ROUTE': FLIGHT_ROUTE,
                "AC_REG": AC_REG,
                'AIRCRAFT_TYPE': AIRCRAFT_TYPE,
                }
            
            record_handling = {
                'ATK_PASSENGER_000': ATK_PASSENGER_000,
                'ATK_000': ATK_000,
                'BLOCK_HOURS': BLOCK_HOURS,
                'FLIGHT_ROUTE': FLIGHT_ROUTE,
                'AC_REG': AC_REG
            }

            record_atc = {
                'ATK_PASSENGER_000': ATK_PASSENGER_000,
                'ATK_000': ATK_000,
                'BLOCK_HOURS': BLOCK_HOURS,
                'AIRCRAFT_TYPE_GROUPING': AIRCRAFT_TYPE_GROUPING,
                'FLIGHT_ROUTE': FLIGHT_ROUTE,
                'ROUNDTRIPROUTE': ROUNDTRIPROUTE,
                'AC_REG': AC_REG,
                'AIRCRAFT_TYPE': AIRCRAFT_TYPE
            }

            try:
                with st.spinner("Meminta prediksi ke API..."):
                    resp = requests.post(
                        API_AFGH_PREDICT, 
                        json={
                                "landing": {"records": [record_landing]},
                                "handling": {"records": [record_handling]},
                                "atc": {"records": [record_atc]},
                            }
                    )
                if resp.status_code == 200:
                    data = resp.json()
                    pred_landing = data["landing"]["predictions"][0]
                    pred_handling = data["handling"]["predictions"][0]
                    pred_atc = data["atc"]["predictions"][0]
                    st.success("Prediksi berhasil ✅")
                    st.metric("Perkiraan biaya Landing ($)", f"{pred_landing:,.2f}")
                    st.metric("Perkiraan biaya Handling ($)", f"{pred_handling:,.2f}")
                    st.metric("Perkiraan biaya ATC ($)", f"{pred_atc:,.2f}")
                    st.json(data)
                    # st.text( f"{actual_landing:,.2f}")
                    # st.text( f"{actual_handling:,.2f}")
                    # st.text( f"{actual_atc:,.2f}")
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Gagal menghubungi API: {e}")

    elif action_menu == "Train":
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
                    resp = requests.post(API_AFGH_TRAIN)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success("Training selesai ✅")
                    st.json(data)
                    st.metric("MAPE Landing (%)", f"{data['landing']['mape_percent']:.2f}")
                    st.metric("RMSE Landing", f"{data['landing']['rmse']:.2f}")

                    st.metric("MAPE Handling (%)", f"{data['handling']['mape_percent']:.2f}")
                    st.metric("RMSE Handling", f"{data['handling']['rmse']:.2f}")

                    st.metric("MAPE ATC (%)", f"{data['atc']['mape_percent']:.2f}")
                    st.metric("RMSE ATC", f"{data['atc']['rmse']:.2f}")

                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Gagal menghubungi API: {e}")