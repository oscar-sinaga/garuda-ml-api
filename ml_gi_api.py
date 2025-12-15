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

# =====================================================================
# KONFIGURASI
# =====================================================================

# Sheet Excel 
EXCEL_PATH = "05. Database RP May 2025 - AC REGISTER.xlsx"
SHEET_NAME = "Raw"

# Model Path
VM_MODEL_PATH = "models/vm_xgb.joblib"
FB_MODEL_PATH = "models/fuel_burn_xgb.joblib"




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


# =====================================================================
# GLOBAL CACHE
# =====================================================================

_vm_artifacts = None

# Global cache untuk model & encoder
_fb_artifacts = None


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



if __name__ == "__main__":

    import uvicorn

    uvicorn.run("ml_gi_api:app", host="0.0.0.0", port=8600, reload=True)