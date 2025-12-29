import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from typing import Union


# =====================================================================
# KONFIGURASI
# =====================================================================

# Sheet Excel 
EXCEL_PATH = "C:/Users/hp/Downloads/acopy/05. Database RP May 2025 - AC REGISTER.xlsx"
SHEET_NAME = "Raw"

# Model Path
VM_MODEL_PATH = "models/vm_xgb.joblib"
FB_MODEL_PATH = "models/fuel_burn_xgb.joblib"
PC_MODEL_PATH = "models/passenger_commission_lgbm.joblib"
R_MODEL_PATH = "models/reservation_xgb.joblib"
OBS_MODEL_PATH = "models/on_board_service_xgb.joblib"
C_MODEL_PATH = "models/catering_xgb.joblib"
MR_MODEL_PATH = "models/maintenance_reserve_xgb.joblib"
COCKPIT_MODEL_PATH = "models/cockpit_crew_xgb.joblib"
CABIN_MODEL_PATH = "models/cabin_crew_xgb.joblib"


# ========================== SELECTED FEATURES ==========================
# VM features
SELECTED_FEATURES_VM = [
    "BLOCK_HOURS", "FLIGHT_HOURS", "FUEL_BURN_IN_LITER",
    "ASK_000", "ATK_PASSENGER_000", "ATK_000",
    "LEASE_AIRCRAFT", "CABIN_CREW_TRAVEL", "FUEL_AIRCRAFT",
    "COCKPIT_CREW_TRAVEL", "CABIN_CREW_PERSON",
    "ROUNDTRIPROUTE", "AIRCRAFT_TYPE", "SERVICE_TYPE",
    "AC_REG", "FLIGHT_ROUTE", "PERIODE",
    "FH_per_BH", "Fuel_per_FH", "ATK_per_ASK"
]


CATEGORICAL_COLS_VM = [
    "AC_REG", "PERIODE", "ROUNDTRIPROUTE",
    "AIRCRAFT_TYPE", "SERVICE_TYPE", "FLIGHT_ROUTE"
]


NUMERICAL_COLS_VM = list(set(SELECTED_FEATURES_VM) - set(CATEGORICAL_COLS_VM))

# FB features
SELECTED_FEATURES_FB = [
    "ROUNDTRIPROUTE",
    "AIRCRAFT_TYPE",
    "AC_REG",
    "SERVICE_TYPE",
    "FLIGHT_ROUTE",           # categorical
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

NUMERICAL_COLS_FB = [c for c in SELECTED_FEATURES_FB if c not in CATEGORICAL_COLS_FB]
TARGET_COL_FB = "FUEL_BURN_LITER"

# PC features
SELECTED_FEATURES_PC = [
    "RPK_000_C_CLASS",
    "RTK_000",
    "RPK_000",
    "RTK_PASSENGER_000",
    "RPK_000_Y_CLASS",
    "ASK_000_C_CLASS",
    "PASSENGER_CARRIED_C_CLASS"
]

TARGET_COL_PC = "PASSENGER_COMMISSION"

# RESERVATION features
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

TARGET_COL_RESERVATION = "RESERVATION"

CATEGORICAL_COLS_RESERVATION = ['SERVICE_TYPE', 
                                'FLIGHT_ROUTE', 
                                'AIRCRAFT_TYPE', 
                                'REGION']

NUMERICAL_COLS_RESERVATION = list(set(SELECTED_FEATURES_RESERVATION) - set(CATEGORICAL_COLS_RESERVATION))

SELECTED_FEATURES_MR_MODEL = [
    'FLIGHT_HOURS',
    'FUEL_BURN_IN_LITER',
    'NUMBER_OF_LANDING',
    'ATK_000',
    'LEASE_AIRCRAFT',
    'AIRCRAFT_TYPE',
    'AC_REG',
    'PERIODE',
    'FH_per_Cycle' # Fitur turunan
]

CATEGORICAL_COLS_MR = ['AC_REG', 'PERIODE', 'AIRCRAFT_TYPE']
NUMERICAL_COLS_MR = list(set(SELECTED_FEATURES_MR_MODEL) - set(CATEGORICAL_COLS_MR))
TARGET_COL_MR = "MAINTENANCE RESERVE"

SELECTED_FEATURES_CREW = [
    'BLOCK_HOURS',          
    'FLIGHT_KILOMETERS',    
    'ASK_000',             
    'NUMBER_OF_LANDING',    
    'AIRCRAFT_TYPE',        
    'SERVICE_TYPE',         
    'PERIODE',
    'ATK_000', 
    'SEAT_OFFERED'            
]

CATEGORICAL_COLS_CREW = ['AIRCRAFT_TYPE', 'SERVICE_TYPE', 'PERIODE']
NUMERICAL_COLS_CREW = list(set(SELECTED_FEATURES_CREW) - set(CATEGORICAL_COLS_CREW))

# ON BOARD SERVICE AND CATERING features

# OBS
SELECTED_FEATURES_OBS = ['PASSENGER_CARRIED',
                         'ATK_PASSENGER_000', 
                         'ASK_000', 
                         'ASK_000_Y_CLASS', 
                         'ATK_000',
                         'CABIN_CREW_PERSON', 
                         'COCKPIT_CREW_PERSON']

TARGET_COL_OBS = ['ON_BOARD_SERVICE']

# CATERING
SELECTED_FEATURES_CATERING = ['PASSENGER_CARRIED', 
                            'ATK_000', 
                            'ATK_PASSENGER_000', 
                            'ASK_000', 
                            'ASK_000_Y_CLASS',
                            'FLIGHT_ROUTE', 
                            'SERVICE_TYPE', 
                            'REGION']

TARGET_COL_CATERING = 'CATERING'

CATEGORICAL_COLS_CATERING = ['FLIGHT_ROUTE', 
                            'SERVICE_TYPE', 
                            'REGION']

NUMERICAL_COLS_CATERING = list(set(SELECTED_FEATURES_CATERING) - set(CATEGORICAL_COLS_CATERING))


# =====================================================================
# MODEL REQUEST FORMAT
# =====================================================================

class VMRecord(BaseModel):
    BLOCK_HOURS: float
    FLIGHT_HOURS: float
    FUEL_BURN_IN_LITER: float
    ASK_000: float
    ATK_PASSENGER_000: float
    ATK_000: float
    LEASE_AIRCRAFT: float
    CABIN_CREW_TRAVEL: float
    FUEL_AIRCRAFT: float
    COCKPIT_CREW_TRAVEL: float
    CABIN_CREW_PERSON: float
    ROUNDTRIPROUTE: str
    AIRCRAFT_TYPE: str
    SERVICE_TYPE: str
    AC_REG: str
    FLIGHT_ROUTE: str
    PERIODE: str

class VMPredictRequest(BaseModel):
    records: List[VMRecord]


class VMPredictResponse(BaseModel):
    predictions: List[float]


class VMTrainResponse(BaseModel):
    mape: float
    mape_percent: float
    rmse: float
    n_train: int
    n_test: int



class FBRecord(BaseModel):
    ROUNDTRIPROUTE: str
    AIRCRAFT_TYPE: str
    AC_REG: str
    SERVICE_TYPE: str
    FLIGHT_ROUTE: str
    CARGO_CARRIED: float
    FREIGHT_CARRIED: float
    ASK_000: float
    ATK_000: float
    ATK_PASSENGER_000: float
    ASK_000_Y_CLASS: float
    ASK_000_C_CLASS: float
    RTK_000: float
    RPK_000: float
    RPK_000_Y_CLASS: float
    RTK_PASSENGER_000: float
    ADMINISTRATION_HO: float


class FBPredictRequest(BaseModel):
    records: List[FBRecord]


class FBPredictResponse(BaseModel):
    predictions: List[float]


class FBTrainResponse(BaseModel):
    mape: float          # dalam desimal, misal 0.05 = 5%
    mape_percent: float  # dalam persen
    rmse: float
    n_train: int
    n_test: int


class PCRecord(BaseModel):
    RPK_000_C_CLASS : float
    RTK_000 : float
    RPK_000 : float
    RTK_PASSENGER_000: float
    RPK_000_Y_CLASS: float
    ASK_000_C_CLASS: float
    PASSENGER_CARRIED_C_CLASS: float

class PCPredictRequest(BaseModel):
    records: List[PCRecord]


class PC_PredictResponse(BaseModel):
    predictions: List[float]


class PC_TrainResponse(BaseModel):
    mape: float          # dalam desimal, misal 0.05 = 5%
    mape_percent: float  # dalam persen
    rmse: float
    n_train: int
    n_test: int

class ReservationRecord(BaseModel):
    PASSENGER_CARRIED : float
    PASSENGER_CARRIED_Y_CLASS : float
    PASSENGER_CARRIED_C_CLASS : float
    CARGO_CARRIED: float
    RPK_000 : float
    RPK_000_Y_CLASS : float
    SEAT_OFFERED : float
    SEAT_OFFERED_Y_CLASS: float
    FLIGHT_ROUTE: str
    SERVICE_TYPE: str
    AIRCRAFT_TYPE: str
    REGION: str

class ReservationPredictRequest(BaseModel):
    records: List[ReservationRecord]

class ReservationPredictResponse(BaseModel):
    predictions: List[float]

class ReservationTrainResponse(BaseModel):
    mape: float          # dalam desimal, misal 0.05 = 5%
    mape_percent: float  # dalam persen
    rmse: float
    n_train: int
    n_test: int


## On Board Service 
class OBSRecord(BaseModel):
    PASSENGER_CARRIED: float
    ATK_PASSENGER_000: float
    ASK_000: float
    ASK_000_Y_CLASS: float
    ATK_000: float
    CABIN_CREW_PERSON: float
    COCKPIT_CREW_PERSON: float

class OBSPredictRequest(BaseModel):
    records: List[OBSRecord]

class OBSPredictResponse(BaseModel):
    predictions: List[float]

class OBSTrainResponse(BaseModel):
    mape: float          # dalam desimal, misal 0.05 = 5%
    mape_percent: float  # dalam persen

# Catering
class CateringRecord(BaseModel):
    PASSENGER_CARRIED: float
    ATK_000: float
    ATK_PASSENGER_000: float
    ASK_000: float
    ASK_000_Y_CLASS: float
    FLIGHT_ROUTE: str
    SERVICE_TYPE: str
    REGION: str

class CateringPredictRequest(BaseModel):
    records: List[CateringRecord]

class CateringPredictResponse(BaseModel):
    predictions: List[float]

class CateringTrainResponse(BaseModel):
    mape: float          # dalam desimal, misal 0.05 = 5%
    mape_percent: float  # dalam persen
      
# Combined OBS and Catering
class OBSCPredictRequest(BaseModel):
    obs: OBSPredictRequest
    catering: CateringPredictRequest

class OBSCPredictResponse(BaseModel):
    obs: OBSPredictResponse
    catering: CateringPredictResponse

class OBSCTrainResponse(BaseModel):
    obs : OBSTrainResponse
    catering : CateringTrainResponse

# Crew
class CrewRecord(BaseModel):
    BLOCK_HOURS: float
    FLIGHT_KILOMETERS: float
    ASK_000: float
    NUMBER_OF_LANDING: float
    ATK_000: float
    SEAT_OFFERED: float
    AIRCRAFT_TYPE: str
    SERVICE_TYPE: str
    PERIODE: str

class CrewPredictRequest(BaseModel):
    records: List[CrewRecord]

class CrewPredictResponse(BaseModel):
    predictions: List[float]

class CrewTrainResponse(BaseModel):
    target: str
    mape: float
    mape_percent: float
    rmse: float
    n_train: int
    n_test: int
      
      
# Maintenance Reserve      
class MRRecord(BaseModel):
    AC_REG: str
    PERIODE: str
    AIRCRAFT_TYPE: str
    FLIGHT_HOURS: float
    FUEL_BURN_IN_LITER: float
    NUMBER_OF_LANDING: float
    ATK_000: float
    LEASE_AIRCRAFT: float

class MRPredictRequest(BaseModel):
    records: List[MRRecord]

class MRPredictResponse(BaseModel):
    predictions: List[float]

class MRTrainResponse(BaseModel):
    mape: float
    mape_percent: float
    rmse: float
    n_train: int
    n_test: int



# =====================================================================
# GLOBAL CACHE
# =====================================================================

_vm_artifacts = None

# Global cache untuk model & encoder
_fb_artifacts = None

_pc_model_artifacts = None

_reservation_model_artifacts = None

_obs_model_artifacts = None
_catering_model_artifacts = None

_mr_artifacts = None

_cockpit_artifacts = None
_cabin_artifacts = None
# =====================================================================
# LOAD ARTIFACTS
# =====================================================================

def load_vm_artifacts():
    global _vm_artifacts
    if _vm_artifacts is not None:
        return _vm_artifacts

    if not os.path.exists(VM_MODEL_PATH):
        raise RuntimeError(f"Model VM tidak ditemukan di {VM_MODEL_PATH}")

    _vm_artifacts = joblib.load(VM_MODEL_PATH)
    return _vm_artifacts


def load_fb_artifacts():
    """Load model & encoder dari disk jika belum ada di cache."""
    global _fb_artifacts
    if _fb_artifacts is not None:
        return _fb_artifacts

    if not os.path.exists(FB_MODEL_PATH):
        raise RuntimeError(
            f"Model belum dilatih. Jalankan endpoint /train dulu. "
            f"File tidak ditemukan: {FB_MODEL_PATH}"
        )

    _fb_artifacts = joblib.load(FB_MODEL_PATH)
    return _fb_artifacts


def load_pc_artifacts():
    """Load model & encoder dari disk jika belum ada di cache."""
    global _pc_model_artifacts
    if _pc_model_artifacts is not None:
        return _pc_model_artifacts

    if not os.path.exists(PC_MODEL_PATH):
        raise RuntimeError(
            f"Model belum dilatih. Jalankan endpoint /train dulu. "
            f"File tidak ditemukan: {PC_MODEL_PATH}"
        )

    _pc_model_artifacts = joblib.load(PC_MODEL_PATH)
    return _pc_model_artifacts

def load_reservation_artifacts():
    """Load model & encoder dari disk jika belum ada di cache."""
    global _reservation_model_artifacts
    if _reservation_model_artifacts is not None:
        return _reservation_model_artifacts

    if not os.path.exists(R_MODEL_PATH):
        raise RuntimeError(
            f"Model belum dilatih. Jalankan endpoint /train dulu. "
            f"File tidak ditemukan: {R_MODEL_PATH}"
        )

    _reservation_model_artifacts = joblib.load(R_MODEL_PATH)
    return _reservation_model_artifacts

def load_obs_artifacts():
    """Load model & encoder dari disk jika belum ada di cache."""
    global _obs_model_artifacts
    if _obs_model_artifacts is not None:
        return _obs_model_artifacts

    if not os.path.exists(OBS_MODEL_PATH):
        raise RuntimeError(
            f"Model belum dilatih. Jalankan endpoint /train dulu. "
            f"File tidak ditemukan: {OBS_MODEL_PATH}"
        )

    _obs_model_artifacts = joblib.load(OBS_MODEL_PATH)
    return _obs_model_artifacts


def load_catering_artifacts():
    """Load model & encoder dari disk jika belum ada di cache."""
    global _catering_model_artifacts
    if _catering_model_artifacts is not None:
        return _catering_model_artifacts

    if not os.path.exists(C_MODEL_PATH):
        raise RuntimeError(
            f"Model belum dilatih. Jalankan endpoint /train dulu. "
            f"File tidak ditemukan: {C_MODEL_PATH}"
        )

    _catering_model_artifacts = joblib.load(C_MODEL_PATH)
    return _catering_model_artifacts



def load_mr_artifacts():
    global _mr_artifacts
    if _mr_artifacts is not None:
        return _mr_artifacts

    if not os.path.exists(MR_MODEL_PATH):
        raise RuntimeError(f"Model MR belum dilatih. Jalankan endpoint /train_mr dulu.")

    _mr_artifacts = joblib.load(MR_MODEL_PATH)
    return _mr_artifacts


def load_cockpit_artifacts():
    global _cockpit_artifacts
    if _cockpit_artifacts is not None: return _cockpit_artifacts
    if not os.path.exists(COCKPIT_MODEL_PATH):
        raise RuntimeError("Model Cockpit belum dilatih.")
    _cockpit_artifacts = joblib.load(COCKPIT_MODEL_PATH)
    return _cockpit_artifacts

def load_cabin_artifacts():
    global _cabin_artifacts
    if _cabin_artifacts is not None: return _cabin_artifacts
    if not os.path.exists(CABIN_MODEL_PATH):
        raise RuntimeError("Model Cabin belum dilatih.")
    _cabin_artifacts = joblib.load(CABIN_MODEL_PATH)
    return _cabin_artifacts
# =====================================================================
# TRAINING FUNCTION
# =====================================================================

def train_vm_model():
    """
    Melatih ulang model Variable Maintenance dari Excel.
    Menggunakan logika preprocessing persis seperti di notebook Oscar.
    """
    if not os.path.exists(EXCEL_PATH):
        raise RuntimeError(f"File Excel VM tidak ditemukan: {EXCEL_PATH}")

    # ============================ Load Data ============================
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, skiprows=1)
    df = df.iloc[:, 1:]   # drop index pertama seperti notebook

    # ============================
    # NORMALISASI NAMA KOLOM
    # ============================

    rename_cols = {
        "BLOCK HOURS": "BLOCK_HOURS",
        "FLIGHT HOURS": "FLIGHT_HOURS",
        "FUEL BURN (IN LITER)": "FUEL_BURN_IN_LITER",
        "ASK (000)": "ASK_000",
        "ATK PASSENGER (000)": "ATK_PASSENGER_000",
        "ATK (000)": "ATK_000",
        "LEASE AIRCRAFT": "LEASE_AIRCRAFT",
        "CABIN CREW TRAVEL": "CABIN_CREW_TRAVEL",
        "FUEL AIRCRAFT": "FUEL_AIRCRAFT",
        "COCKPIT CREW TRAVEL": "COCKPIT_CREW_TRAVEL",
        "CABIN CREW PERSON": "CABIN_CREW_PERSON",
        "ROUNDTRIPROUTE": "ROUNDTRIPROUTE",
        "AIRCRAFT TYPE": "AIRCRAFT_TYPE",
        "SERVICE TYPE": "SERVICE_TYPE",
        "AC REG": "AC_REG",
        "FLIGHT ROUTE": "FLIGHT_ROUTE",
        "PERIODE": "PERIODE"
    }

    df.rename(columns=rename_cols, inplace=True)


    # Drop zero VM
    df = df[df['VARIABLE MAINTENANCE'] >= 0].copy()

    df1 = df[
        (df['FUEL_BURN_IN_LITER'] != 0) &
        (df['FLIGHT_HOURS'] != 0) &
        (df['VARIABLE MAINTENANCE'] != 0)
    ].copy()

    # ====================== Group by AC REG & PERIODE ======================
    df_group = df1.groupby(["AC_REG", "PERIODE"]).agg({
                "VARIABLE MAINTENANCE": "sum",
                "BLOCK_HOURS": "sum",
                "FLIGHT_HOURS": "sum",
                "FUEL_BURN_IN_LITER": "sum",
                "FUEL_AIRCRAFT": "sum",
                "ASK_000": "sum",
                "ATK_PASSENGER_000": "sum",
                "ATK_000": "sum",
                "LEASE_AIRCRAFT": "mean",
                "CABIN_CREW_TRAVEL": "sum",
                "COCKPIT_CREW_TRAVEL": "sum",
                "CABIN_CREW_PERSON": "mean",
                "ROUNDTRIPROUTE": "nunique",
                "FLIGHT_ROUTE": "nunique",
                "SERVICE_TYPE": "first",
                "AIRCRAFT_TYPE": "first",
            }).reset_index()


    # Derived features
    df_group["FH_per_BH"] = df_group["FLIGHT_HOURS"] / df_group["BLOCK_HOURS"].replace(0, np.nan)
    df_group["Fuel_per_FH"] = df_group["FUEL_BURN_IN_LITER"] / df_group["FLIGHT_HOURS"].replace(0, np.nan)
    df_group["ATK_per_ASK"] = df_group["ATK_000"] / df_group["ASK_000"].replace(0, np.nan)


    

    X = df_group[SELECTED_FEATURES_VM].copy()
    y = df_group["VARIABLE MAINTENANCE"].copy()

    # =================== Split by AC REG (GroupSplit) ====================
    splitter = GroupShuffleSplit(test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(df_group, groups=df_group["AC_REG"]))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # =========================== ENCODER ================================
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(X_train[CATEGORICAL_COLS_VM])

    X_train_enc = encoder.transform(X_train[CATEGORICAL_COLS_VM])
    X_test_enc = encoder.transform(X_test[CATEGORICAL_COLS_VM])

    enc_cols = encoder.get_feature_names_out(CATEGORICAL_COLS_VM)

    X_train_enc_df = pd.DataFrame(X_train_enc, columns=enc_cols, index=X_train.index)
    X_test_enc_df = pd.DataFrame(X_test_enc, columns=enc_cols, index=X_test.index)

    X_train_final = pd.concat([X_train[NUMERICAL_COLS_VM], X_train_enc_df], axis=1)
    X_test_final = pd.concat([X_test[NUMERICAL_COLS_VM], X_test_enc_df], axis=1)

    # =========================== TRAIN MODEL =============================
    model = XGBRegressor(
        n_estimators=500,
        max_depth=10,
        learning_rate=0.05,
        gamma=0.1,
        objective="reg:squarederror"
    )

    model.fit(X_train_final, y_train)
    y_pred = model.predict(X_test_final)

    # =========================== METRICS =============================
    mape = mean_absolute_percentage_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # ====================== SAVE ARTIFACTS =========================
    artifacts = {
        "model": model,
        "encoder": encoder,
        "categorical_cols": CATEGORICAL_COLS_VM,
        "numeric_cols": NUMERICAL_COLS_VM,
        "selected_features": SELECTED_FEATURES_VM
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump(artifacts, VM_MODEL_PATH)

    return {
        "mape": float(mape),
        "mape_percent": float(mape * 100),
        "rmse": float(rmse),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test))
    }


def train_fb_model():
    """Train XGBRegressor + OneHotEncoder, simpan artifacts, dan return metrics."""
    global _fb_artifacts

    def load_training_data() -> pd.DataFrame:
        """Baca Excel, rename kolom, pilih fitur dan target, drop NaN."""
        if not os.path.exists(EXCEL_PATH):
            raise RuntimeError(f"File Excel tidak ditemukan: {EXCEL_PATH}")

        df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, skiprows=1)
        df = df.iloc[:, 1:]  # buang kolom index pertama
        df = df[(df['FUEL BURN (IN LITER)']!=0) & (df['FLIGHT HOURS']!=0)].copy()

        # Mapping nama kolom Excel -> nama Pythonic
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

        df = df.rename(columns=RENAME_MAP)

        cols_needed = SELECTED_FEATURES_FB + [TARGET_COL_FB]
        missing = [c for c in cols_needed if c not in df.columns]
        if missing:
            raise RuntimeError(f"Kolom berikut tidak ditemukan di dataset: {missing}")

        df1 = df[cols_needed].dropna()
        
        return df1

    df1 = load_training_data()
    X = df1[SELECTED_FEATURES_FB].copy()
    y = df1[TARGET_COL_FB].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(X_train[CATEGORICAL_COLS_FB])

    X_train_cat = encoder.transform(X_train[CATEGORICAL_COLS_FB])
    X_test_cat = encoder.transform(X_test[CATEGORICAL_COLS_FB])

    encoded_cols = encoder.get_feature_names_out(CATEGORICAL_COLS_FB)

    X_train_cat_df = pd.DataFrame(
        X_train_cat, columns=encoded_cols, index=X_train.index
    )
    X_test_cat_df = pd.DataFrame(
        X_test_cat, columns=encoded_cols, index=X_test.index
    )

    X_train_final = pd.concat([X_train[NUMERICAL_COLS_FB], X_train_cat_df], axis=1)
    X_test_final = pd.concat([X_test[NUMERICAL_COLS_FB], X_test_cat_df], axis=1)

    model = XGBRegressor(
        n_estimators=500,
        max_depth=15,
        learning_rate=0.05,
        gamma=0.01,
        objective="reg:squarederror",
        n_jobs=-1,
    )
    model.fit(X_train_final, y_train)

    y_pred = model.predict(X_test_final)

    mape = mean_absolute_percentage_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))


    artifacts = {
        "model": model,
        "encoder": encoder,
        "categorical_cols": CATEGORICAL_COLS_FB,
        "numeric_cols": NUMERICAL_COLS_FB,
        "selected_features": SELECTED_FEATURES_FB,
    }

    os.makedirs(os.path.dirname(FB_MODEL_PATH), exist_ok=True)
    joblib.dump(artifacts, FB_MODEL_PATH)

    _fb_artifacts = artifacts  # cache di memori

    return {
        "mape": float(mape),
        "mape_percent": float(mape * 100.0),
        "rmse": float(rmse),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }




def train_pc_model():
    """Train, simpan artifacts, dan return metrics."""
    global _pc_model_artifacts

    def load_training_data() -> pd.DataFrame:
        if not os.path.exists(EXCEL_PATH):
            raise RuntimeError(f"File Excel tidak ditemukan: {EXCEL_PATH}")

        df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, skiprows=1)
        df = df.iloc[:, 1:]  # buang kolom index pertama

        # Mapping nama kolom Excel -> nama Pythonic
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

        df = df.rename(columns=RENAME_MAP)

        ### REMOVE ZEROES
        df1 = df[(df['FLIGHT_KILOMETERS']!=0) & (df['SEAT_OFFERED']!=0) & 
                (df['PASSENGER_CARRIED']!=0) & (df['PASSENGER_COMMISSION']!=0) &
                (df['BLOCK_HOURS']!=0) & (df['FUEL_AIRCRAFT']>0)].copy()
        return df1

    df1 = load_training_data()
    X = df1[SELECTED_FEATURES_PC].copy()
    y = df1[TARGET_COL_PC].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.01,
    num_leaves=255,
    max_depth=-1,
    min_data_in_leaf=20
)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mape = mean_absolute_percentage_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    artifacts = {
        "model": model,
        "selected_features": SELECTED_FEATURES_PC,
    }

    os.makedirs(os.path.dirname(PC_MODEL_PATH), exist_ok=True)
    joblib.dump(artifacts, PC_MODEL_PATH)

    _pc_model_artifacts = artifacts  # cache di memori

    return {
        "mape": float(mape),
        "mape_percent": float(mape * 100.0),
        "rmse": float(rmse),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }


def train_reservation_model():
     """Train, simpan artifacts, dan return metrics."""
     global _reservation_model_artifacts
     
     def load_training_data() -> pd.DataFrame:
        if not os.path.exists(EXCEL_PATH):
            raise RuntimeError(f"File Excel tidak ditemukan: {EXCEL_PATH}")

        df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, skiprows=1)
        df = df.iloc[:, 1:]  # buang kolom index pertama

        RENAME_MAP = {
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

        df = df.rename(columns=RENAME_MAP)

        ### REMOVE ZEROES
        df1 = df[(df['CARGO_CARRIED']!=0) & (df['FREIGHT_CARRIED']!=0) & (df['PASSENGER_CARRIED'] != 0) &
                (df['BLOCK_HOURS']!=0) & (df['RESERVATION']!=0)].copy()

        return df1
     

     df1 = load_training_data()
     
     X = df1[SELECTED_FEATURES_RESERVATION].copy()
     y = df1[TARGET_COL_RESERVATION].copy()

     X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
     

     encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
     encoder.fit(X_train[CATEGORICAL_COLS_RESERVATION])
     
     X_train_cat = encoder.transform(X_train[CATEGORICAL_COLS_RESERVATION])
     X_test_cat = encoder.transform(X_test[CATEGORICAL_COLS_RESERVATION])
     
     encoded_cols = encoder.get_feature_names_out(CATEGORICAL_COLS_RESERVATION)
     
     X_train_cat_df = pd.DataFrame(
        X_train_cat, columns=encoded_cols, index=X_train.index
     )
     
     X_test_cat_df = pd.DataFrame(
        X_test_cat, columns=encoded_cols, index=X_test.index
        )
     
     X_train_final = pd.concat([X_train[NUMERICAL_COLS_RESERVATION], X_train_cat_df], axis=1)
     X_test_final = pd.concat([X_test[NUMERICAL_COLS_RESERVATION], X_test_cat_df], axis=1)

     model = XGBRegressor(n_estimators=1500, learning_rate=0.01, objective="reg:squarederror")
     model.fit(X_train_final, y_train)


     y_pred = model.predict(X_test_final)
     
     mape = mean_absolute_percentage_error(y_test, y_pred)
     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
     
     
     artifacts = {
        "model": model,
        "encoder": encoder,
        "categorical_cols": CATEGORICAL_COLS_RESERVATION,
        "numeric_cols": NUMERICAL_COLS_RESERVATION,
        "selected_features": SELECTED_FEATURES_RESERVATION,
    }
     
     os.makedirs(os.path.dirname(R_MODEL_PATH), exist_ok=True)
     
     joblib.dump(artifacts, R_MODEL_PATH)
     
     _reservation_model_artifacts = artifacts  # cache di memori
     
     return {
        "mape": float(mape),
        "mape_percent": float(mape * 100.0),
        "rmse": float(rmse),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }


def train_obs_model():
    """Train, simpan artifacts, dan return metrics."""
    global _obs_model_artifacts
     
    def load_training_data() -> pd.DataFrame:
        if not os.path.exists(EXCEL_PATH):
            raise RuntimeError(f"File Excel tidak ditemukan: {EXCEL_PATH}")

        df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, skiprows=1)
        df = df.iloc[:, 1:]  # buang kolom index pertama

        RENAME_MAP = {
            # On Board Service
            'PASSENGER CARRIED': 'PASSENGER_CARRIED',
            'ATK PASSENGER (000)': 'ATK_PASSENGER_000', 
            'ASK (000)': 'ASK_000', 
            'ASK (000) Y CLASS': 'ASK_000_Y_CLASS', 
            'ATK (000)': 'ATK_000',
            'CABIN CREW PERSON': 'CABIN_CREW_PERSON', 
            'COCKPIT CREW PERSON': 'COCKPIT_CREW_PERSON',
            'BLOCK HOURS': 'BLOCK_HOURS',
            'ON BOARD SERVICE': 'ON_BOARD_SERVICE'
        }

        df = df.rename(columns=RENAME_MAP)

        ### REMOVE ZEROES
        df1 = df[(df['PASSENGER_CARRIED'] != 0) & 
                 (df['BLOCK_HOURS']!=0) & 
        (df['ON_BOARD_SERVICE']!=0) & (df['CATERING']!=0)].copy()
        
        return df1
    
    df1 = load_training_data()
    X = df1[SELECTED_FEATURES_OBS].copy()
    y = df1[TARGET_COL_OBS].copy()

    X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42)
    
    model = XGBRegressor(n_estimators=2000, learning_rate=0.5, objective="reg:squarederror")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    mape = mean_absolute_percentage_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    artifacts = {
        "model": model,
        "selected_features": SELECTED_FEATURES_OBS,
    }

    os.makedirs(os.path.dirname(OBS_MODEL_PATH), exist_ok=True)
    
    joblib.dump(artifacts, OBS_MODEL_PATH)
    
    _obs_model_artifacts = artifacts  # cache di memori
    
    return {
        "mape": float(mape),
        "mape_percent": float(mape * 100.0),
        "rmse": float(rmse),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }


def train_catering_model():
    """Train, simpan artifacts, dan return metrics."""
    global _catering_model_artifacts
     
    def load_training_data() -> pd.DataFrame:
        if not os.path.exists(EXCEL_PATH):
            raise RuntimeError(f"File Excel tidak ditemukan: {EXCEL_PATH}")

        df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, skiprows=1)
        df = df.iloc[:, 1:]  # buang kolom index pertama  

        RENAME_MAP= {
            # Catering
            'PASSENGER CARRIED': 'PASSENGER_CARRIED', 
            'ATK (000)': 'ATK_000', 
            'ATK PASSENGER (000)': 'ATK_PASSENGER_000', 
            'ASK (000)': 'ASK_000', 
            'ASK (000) Y CLASS': 'ASK_000_Y_CLASS',
            'FLIGHT ROUTE': 'FLIGHT_ROUTE', 
            'SERVICE TYPE': 'SERVICE_TYPE', 
            'Region': 'REGION',
            'BLOCK HOURS': 'BLOCK_HOURS',
            'ON BOARD SERVICE': 'ON_BOARD_SERVICE'

        }

        df = df.rename(columns=RENAME_MAP)

        ### REMOVE ZEROES
        df1 = df[(df['PASSENGER_CARRIED'] != 0) & (df['BLOCK_HOURS']!=0) & 
        (df['ON_BOARD_SERVICE']!=0) & (df['CATERING']!=0)].copy()


        return df1
    
    df1 = load_training_data()
    X = df1[SELECTED_FEATURES_CATERING].copy()
    y = df1['CATERING'].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=10)
    
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(X_train[CATEGORICAL_COLS_CATERING])

    X_train_encoded = encoder.transform(X_train[CATEGORICAL_COLS_CATERING])
    X_test_encoded = encoder.transform(X_test[CATEGORICAL_COLS_CATERING])

    encoded_cols = encoder.get_feature_names_out(CATEGORICAL_COLS_CATERING)

    X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoded_cols, index=X_train.index)
    X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoded_cols, index=X_test.index)

    X_train_final = pd.concat([X_train[NUMERICAL_COLS_CATERING], X_train_encoded], axis=1)
    X_test_final = pd.concat([X_test[NUMERICAL_COLS_CATERING], X_test_encoded], axis=1)

    model = XGBRegressor(n_estimators=2000, learning_rate=0.5, objective="reg:squarederror")
    model.fit(X_train_final, y_train)

    y_pred = model.predict(X_test_final)

    mape = mean_absolute_percentage_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    
    artifacts = {
        "model": model,
        "encoder": encoder,
        "categorical_cols": CATEGORICAL_COLS_CATERING,
        "numeric_cols": NUMERICAL_COLS_CATERING,
        "selected_features": SELECTED_FEATURES_CATERING,
    }

    os.makedirs(os.path.dirname(C_MODEL_PATH), exist_ok=True)
    
    joblib.dump(artifacts, C_MODEL_PATH)
    
    _catering_model_artifacts = artifacts  # cache di memori
    
    return {
        "mape": float(mape),
        "mape_percent": float(mape * 100.0),
        "rmse": float(rmse),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }



# ... (Di bawah fungsi train_reservation_model) ...

def train_mr_model():
    global _mr_artifacts
    
    if not os.path.exists(EXCEL_PATH):
        raise RuntimeError(f"File Excel tidak ditemukan: {EXCEL_PATH}")

    # 1. Load Data
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, skiprows=1)
    df = df.iloc[:, 1:] 
    
    # 2. Filter Data (Sesuai snippet pandas kamu)
    # Pastikan kolom target ada dan valid
    if 'MAINTENANCE RESERVE' not in df.columns:
         raise RuntimeError("Kolom 'MAINTENANCE RESERVE' tidak ada di Excel.")

    df = df[df['MAINTENANCE RESERVE'] >= 0].copy()
    
    df1 = df[
        (df['FUEL BURN (IN LITER)'] != 0) &
        (df['FLIGHT HOURS'] != 0) &
        (df['MAINTENANCE RESERVE'] != 0)
    ].copy()

    # 3. Grouping (Logika inti dari snippet kamu)
    # Kita group by nama kolom asli di Excel
    df_group = df1.groupby(["AC REG", "PERIODE"]).agg({
        "MAINTENANCE RESERVE": "sum",
        "FLIGHT HOURS": "sum",
        "NUMBER OF LANDING": "sum",
        "FUEL BURN (IN LITER)": "sum",
        "ATK (000)": "sum",
        "LEASE AIRCRAFT": "mean",
        "SERVICE TYPE": "first", 
        "AIRCRAFT TYPE": "first",
    }).reset_index()

    # 4. Feature Engineering
    # Menghindari pembagian dengan nol
    df_group['FH_per_Cycle'] = df_group.apply(
        lambda x: x['FLIGHT HOURS'] / x['NUMBER OF LANDING'] if x['NUMBER OF LANDING'] > 0 else 0, 
        axis=1
    )
    df_group = df_group.fillna(0)

    # 5. Rename Columns agar sesuai Pydantic (Snake Case)
    rename_map = {
        "AC REG": "AC_REG",
        "AIRCRAFT TYPE": "AIRCRAFT_TYPE",
        "FLIGHT HOURS": "FLIGHT_HOURS",
        "FUEL BURN (IN LITER)": "FUEL_BURN_IN_LITER",
        "NUMBER OF LANDING": "NUMBER_OF_LANDING",
        "ATK (000)": "ATK_000",
        "LEASE AIRCRAFT": "LEASE_AIRCRAFT",
        "MAINTENANCE RESERVE": "MAINTENANCE RESERVE"
    }
    df_group.rename(columns=rename_map, inplace=True)

    # 6. Split Data (GroupShuffleSplit)
    X = df_group[SELECTED_FEATURES_MR_MODEL].copy()
    y = df_group[TARGET_COL_MR].copy()

    splitter = GroupShuffleSplit(test_size=0.2, random_state=42)
    # Split berdasarkan AC_REG agar data pesawat yang sama tidak bocor
    train_idx, test_idx = next(splitter.split(df_group, groups=df_group["AC_REG"]))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # 7. Encoding
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(X_train[CATEGORICAL_COLS_MR])
    
    X_train_enc = encoder.transform(X_train[CATEGORICAL_COLS_MR])
    X_test_enc = encoder.transform(X_test[CATEGORICAL_COLS_MR])
    
    encoded_cols = encoder.get_feature_names_out(CATEGORICAL_COLS_MR)
    
    # Gabungkan Numeric + Encoded
    X_train_final = pd.concat([
        X_train[NUMERICAL_COLS_MR].reset_index(drop=True), 
        pd.DataFrame(X_train_enc, columns=encoded_cols, index=X_train.index).reset_index(drop=True)
    ], axis=1)
    
    X_test_final = pd.concat([
        X_test[NUMERICAL_COLS_MR].reset_index(drop=True), 
        pd.DataFrame(X_test_enc, columns=encoded_cols, index=X_test.index).reset_index(drop=True)
    ], axis=1)

    # 8. Train XGBoost
    model = XGBRegressor(
        n_estimators=500, 
        max_depth=6, 
        learning_rate=0.05, 
        gamma=0.1, 
        objective="reg:squarederror"
    )
    model.fit(X_train_final, y_train)
    
    # 9. Evaluate
    y_pred = model.predict(X_test_final)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # 10. Save Artifacts
    artifacts = {
        "model": model,
        "encoder": encoder,
        "categorical_cols": CATEGORICAL_COLS_MR,
        "numeric_cols": NUMERICAL_COLS_MR
    }
    
    os.makedirs(os.path.dirname(MR_MODEL_PATH), exist_ok=True)
    joblib.dump(artifacts, MR_MODEL_PATH)
    _mr_artifacts = artifacts

    return {
        "mape": float(mape),
        "mape_percent": float(mape * 100),
        "rmse": float(rmse),
        "n_train": len(X_train),
        "n_test": len(X_test)
    }


def _train_generic_crew_model(target_col_excel, model_save_path):
    """
    Fungsi internal untuk melatih model crew (Cockpit atau Cabin).
    """
    if not os.path.exists(EXCEL_PATH):
        raise RuntimeError(f"File Excel tidak ditemukan: {EXCEL_PATH}")

    # 1. Load Data
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, skiprows=1)
    df = df.iloc[:, 1:]
    
    # 2. Rename & Filter
    RENAME_MAP = {
        "BLOCK HOURS": "BLOCK_HOURS",
        "FLIGHT KILOMETERS": "FLIGHT_KILOMETERS",
        "ASK (000)": "ASK_000",
        "NUMBER OF LANDING": "NUMBER_OF_LANDING",
        "AIRCRAFT TYPE": "AIRCRAFT_TYPE",
        "SERVICE TYPE": "SERVICE_TYPE",
        "PERIODE": "PERIODE",
        "ATK (000)": "ATK_000",
        "SEAT OFFERED": "SEAT_OFFERED",
        "COCKPIT CREW TRAVEL": "COCKPIT_CREW_TRAVEL",
        "CABIN CREW TRAVEL": "CABIN_CREW_TRAVEL"
    }
    df.rename(columns=RENAME_MAP, inplace=True)

    # Pastikan target ada
    target_py = RENAME_MAP.get(target_col_excel, target_col_excel)
    
    # Filter sesuai snippet kamu
    df = df[
        (df['BLOCK_HOURS'] > 0) & 
        (df['ASK_000'] > 0) & 
        (df[target_py] > 0) # Kita ambil yang > 0 untuk training agar akurat
    ].copy()

    X = df[SELECTED_FEATURES_CREW].copy()
    y = df[target_py].copy()

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Encoding
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(X_train[CATEGORICAL_COLS_CREW])
    
    X_train_enc = encoder.transform(X_train[CATEGORICAL_COLS_CREW])
    X_test_enc = encoder.transform(X_test[CATEGORICAL_COLS_CREW])
    enc_cols = encoder.get_feature_names_out(CATEGORICAL_COLS_CREW)

    X_train_final = pd.concat([
        X_train[NUMERICAL_COLS_CREW].reset_index(drop=True),
        pd.DataFrame(X_train_enc, columns=enc_cols).reset_index(drop=True)
    ], axis=1)

    X_test_final = pd.concat([
        X_test[NUMERICAL_COLS_CREW].reset_index(drop=True),
        pd.DataFrame(X_test_enc, columns=enc_cols).reset_index(drop=True)
    ], axis=1)

    # 5. Training (Sesuai parameter snippet kamu)
    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        n_jobs=-1,
        objective="reg:absoluteerror" # Mengikuti snippet
    )
    model.fit(X_train_final, y_train)

    # 6. Eval & Save
    y_pred = model.predict(X_test_final)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    artifacts = {
        "model": model,
        "encoder": encoder,
        "categorical_cols": CATEGORICAL_COLS_CREW,
        "numeric_cols": NUMERICAL_COLS_CREW
    }
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(artifacts, model_save_path)

    return {
        "target": target_py,
        "mape": float(mape),
        "mape_percent": float(mape * 100),
        "rmse": float(rmse),
        "n_train": len(X_train),
        "n_test": len(X_test)
    }

# Wrapper functions
def train_cockpit_model():
    global _cockpit_artifacts
    res = _train_generic_crew_model("COCKPIT CREW TRAVEL", COCKPIT_MODEL_PATH)
    _cockpit_artifacts = joblib.load(COCKPIT_MODEL_PATH)
    return res

def train_cabin_model():
    global _cabin_artifacts
    res = _train_generic_crew_model("CABIN CREW TRAVEL", CABIN_MODEL_PATH)
    _cabin_artifacts = joblib.load(CABIN_MODEL_PATH)
    return res


# =====================================================================
# FASTAPI ENDPOINTS
# =====================================================================

app = FastAPI(title="Aircraft Cost Prediction API", version="1.0")


# =======================
# PREDICT
# =======================

@app.post("/predict_vm", response_model=VMPredictResponse)
def predict_vm(req: VMPredictRequest):
    try:
        artifacts = load_vm_artifacts()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    model = artifacts["model"]
    encoder = artifacts["encoder"]
    categorical_cols = artifacts["categorical_cols"]
    numeric_cols = artifacts["numeric_cols"]


    df = pd.DataFrame([r.dict() for r in req.records])

    # Derived features
    df["FH_per_BH"] = df["FLIGHT_HOURS"] / df["BLOCK_HOURS"].replace(0, np.nan)
    df["Fuel_per_FH"] = df["FUEL_BURN_IN_LITER"] / df["FLIGHT_HOURS"].replace(0, np.nan)
    df["ATK_per_ASK"] = df["ATK_000"] / df["ASK_000"].replace(0, np.nan)
    df = df.fillna(0)
    
    df_cat = df[categorical_cols]
    df_num = df[numeric_cols]

    df_cat_enc = encoder.transform(df_cat)
    enc_cols = encoder.get_feature_names_out(categorical_cols)

    df_cat_enc_df = pd.DataFrame(df_cat_enc, columns=enc_cols, index=df.index)

    X_final = pd.concat([df_num, df_cat_enc_df], axis=1)

    preds = model.predict(X_final)
    preds = [float(p) for p in preds]

    return VMPredictResponse(predictions=preds)


@app.post("/predict_fb", response_model=FBPredictResponse)
def predict_fb(req: FBPredictRequest):
    """Prediksi fuel burn (dalam liter) dari fitur input."""
    try:
        artifacts = load_fb_artifacts()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    model = artifacts["model"]
    encoder: OneHotEncoder = artifacts["encoder"]

    # Pydantic -> DataFrame
    data = pd.DataFrame([r.dict() for r in req.records])

    # Pastikan semua fitur ada
    missing = [c for c in SELECTED_FEATURES_FB if c not in data.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Fitur berikut hilang di request: {missing}",
        )

    X_cat = data[CATEGORICAL_COLS_FB]
    X_num = data[NUMERICAL_COLS_FB]

    X_cat_enc = encoder.transform(X_cat)
    encoded_cols = encoder.get_feature_names_out(CATEGORICAL_COLS_FB)
    X_cat_enc_df = pd.DataFrame(X_cat_enc, columns=encoded_cols, index=data.index)

    X_final = pd.concat([X_num, X_cat_enc_df], axis=1)

    preds = model.predict(X_final)
    preds = [float(p) for p in preds]

    return FBPredictResponse(predictions=preds)

@app.post("/predict_pc", response_model=PC_PredictResponse)
def predict_pc(req: PCPredictRequest):
    """Prediksi fuel burn (dalam liter) dari fitur input."""
    try:
        artifacts = load_pc_artifacts()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    model = artifacts["model"]

     # Pydantic -> DataFrame
    data = pd.DataFrame([r.dict() for r in req.records])

    # Pastikan semua fitur ada
    missing = [c for c in SELECTED_FEATURES_PC if c not in data.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Fitur berikut hilang di request: {missing}",
        )

    preds = model.predict(data)
    preds = [float(p) for p in preds]

    return PC_PredictResponse(predictions=preds)


@app.post("/predict_reservation", response_model=ReservationPredictResponse)
def predict_reservation(req: ReservationPredictRequest):
    try:
        artifacts = load_reservation_artifacts()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    model = artifacts["model"]
    encoder = artifacts["encoder"]
    categorical_cols = artifacts["categorical_cols"]
    numeric_cols = artifacts["numeric_cols"]

     # Pydantic -> DataFrame
    df = pd.DataFrame([r.dict() for r in req.records])

    # Pastikan semua fitur ada
    missing = [c for c in SELECTED_FEATURES_RESERVATION if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Fitur berikut hilang di request: {missing}",
        )
    df_cat = df[categorical_cols]
    df_num = df[numeric_cols]

    df_cat_enc = encoder.transform(df_cat)
    enc_cols = encoder.get_feature_names_out(categorical_cols)

    df_cat_enc_df = pd.DataFrame(df_cat_enc, columns=enc_cols, index=df.index)

    X_final = pd.concat([df_num, df_cat_enc_df], axis=1)


    preds = model.predict(X_final)
    preds = [float(p) for p in preds]

    return ReservationPredictResponse(predictions= preds)


@app.post("/predict_obsc", response_model=OBSCPredictResponse)
def predict_obsc(req: OBSCPredictRequest):
    try:
        artifacts_obs = load_obs_artifacts()
        artifacts_catering = load_catering_artifacts()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    model_obs = artifacts_obs["model"]

    model_catering = artifacts_catering["model"]
    encoder_catering = artifacts_catering["encoder"]
    categorical_cols_catering = artifacts_catering["categorical_cols"]
    numeric_cols_catering = artifacts_catering["numeric_cols"]


    # Pydantic -> DataFrame
    data_obs = pd.DataFrame([r.dict() for r in req.obs.records])
    data_catering =pd.DataFrame([r.dict() for r in req.catering.records])

    # Pastikan semua fitur ada
    missing = [c for c in SELECTED_FEATURES_OBS if c not in data_obs.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Fitur berikut hilang di request: {missing}",
        )
    
    missing = [c for c in SELECTED_FEATURES_CATERING if c not in data_catering.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Fitur berikut hilang di request: {missing}",
        )
    
    # OBS
    preds_obs = model_obs.predict(data_obs)
    preds_obs = [float(p) for p in preds_obs]

    # CATERING
    df_cat = data_catering[categorical_cols_catering]
    df_num = data_catering[numeric_cols_catering]

    df_cat_enc = encoder_catering.transform(df_cat)
    enc_cols = encoder_catering.get_feature_names_out(categorical_cols_catering)

    df_cat_enc_df = pd.DataFrame(df_cat_enc, columns=enc_cols, index=data_catering.index)

    X_final = pd.concat([df_num, df_cat_enc_df], axis=1)

    preds_catering = model_catering.predict(X_final)
    preds_catering = [float(p) for p in preds_catering]

    return {
        "obs": OBSPredictResponse(predictions=preds_obs),
        "catering": CateringPredictResponse(predictions=preds_catering)
    }




# =======================
# TRAIN
# =======================

@app.post("/train_vm", response_model=VMTrainResponse)
def train_vm():
    """
    Latih ulang model VM dari Excel.
    """
    try:
        metrics = train_vm_model()
        return VMTrainResponse(**metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    



@app.post("/train_fb", response_model=FBTrainResponse)
def train_fb():
    """Latih ulang model dari file Excel."""
    try:
        metrics = train_fb_model()
        return FBTrainResponse(**metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train_pc", response_model=PC_TrainResponse)
def train_pc():
    """Latih ulang model dari file Excel."""
    try:
        metrics = train_pc_model()
        return PC_TrainResponse(**metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train_reservation", response_model=ReservationTrainResponse)
def train_reservation():
    """Latih ulang model dari file Excel."""
    try:
        metrics = train_reservation_model()
        return ReservationTrainResponse(**metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train_obsc", response_model=OBSCTrainResponse)
def train_obsc():
    """Latih ulang model dari file Excel."""
    try:
        metrics_obs = train_obs_model()
        metrics_catering = train_catering_model()

        return {"obs": OBSTrainResponse(**metrics_obs),               
                "catering": CateringTrainResponse(**metrics_catering)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =======================
# MAINTENANCE RESERVE ENDPOINTS
# =======================

@app.post("/train_mr", response_model=MRTrainResponse)
def train_mr():
    try:
        metrics = train_mr_model()
        return MRTrainResponse(**metrics)
    except Exception as e:
        print(f"Error Training MR: {e}") # Print ke terminal untuk debug
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_mr", response_model=MRPredictResponse)
def predict_mr(req: MRPredictRequest):
    try:
        artifacts = load_mr_artifacts()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    model = artifacts["model"]
    encoder = artifacts["encoder"]
    cat_cols = artifacts["categorical_cols"]
    num_cols = artifacts["numeric_cols"]
    
    # Convert request -> DataFrame
    df = pd.DataFrame([r.dict() for r in req.records])
    
    # Calculate derived feature (HARUS SAMA dengan logic training)
    df['FH_per_Cycle'] = df.apply(
        lambda x: x['FLIGHT_HOURS'] / x['NUMBER_OF_LANDING'] if x['NUMBER_OF_LANDING'] > 0 else 0, 
        axis=1
    )
    
    # Preprocessing
    df_cat = df[cat_cols]
    df_num = df[num_cols]
    
    df_cat_enc = encoder.transform(df_cat)
    encoded_cols = encoder.get_feature_names_out(cat_cols)
    
    X_final = pd.concat([
        df_num.reset_index(drop=True),
        pd.DataFrame(df_cat_enc, columns=encoded_cols)
    ], axis=1)
    
    preds = model.predict(X_final)
    return MRPredictResponse(predictions=[float(p) for p in preds])


# CREW TRAVEL ENDPOINTS
# =======================

@app.post("/train_cockpit", response_model=CrewTrainResponse)
def endpoint_train_cockpit():
    try:
        return CrewTrainResponse(**train_cockpit_model())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train_cabin", response_model=CrewTrainResponse)
def endpoint_train_cabin():
    try:
        return CrewTrainResponse(**train_cabin_model())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _predict_crew_generic(req: CrewPredictRequest, artifacts):
    model = artifacts["model"]
    encoder = artifacts["encoder"]
    cat_cols = artifacts["categorical_cols"]
    num_cols = artifacts["numeric_cols"]
    
    df = pd.DataFrame([r.dict() for r in req.records])
    
    df_cat = df[cat_cols]
    df_num = df[num_cols]
    
    df_cat_enc = encoder.transform(df_cat)
    encoded_cols = encoder.get_feature_names_out(cat_cols)
    
    X_final = pd.concat([
        df_num.reset_index(drop=True),
        pd.DataFrame(df_cat_enc, columns=encoded_cols)
    ], axis=1)
    
    preds = model.predict(X_final)
    return [float(p) for p in preds]

@app.post("/predict_cockpit", response_model=CrewPredictResponse)
def predict_cockpit(req: CrewPredictRequest):
    try:
        artifacts = load_cockpit_artifacts()
        preds = _predict_crew_generic(req, artifacts)
        return CrewPredictResponse(predictions=preds)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_cabin", response_model=CrewPredictResponse)
def predict_cabin(req: CrewPredictRequest):
    try:
        artifacts = load_cabin_artifacts()
        preds = _predict_crew_generic(req, artifacts)
        return CrewPredictResponse(predictions=preds)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":

    import uvicorn

    uvicorn.run("ml_gi_api:app", host="0.0.0.0", port=8600, reload=True)