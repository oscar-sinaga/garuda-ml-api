#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
API Fuel Burn (XGBoost + OneHotEncoder)

Endpoint:
- POST /train   -> latih model dari file Excel dan simpan ke disk
- POST /predict -> prediksi FUEL_BURN_LITER dari input fitur

Cara jalanin:
1) pip install fastapi uvicorn pandas scikit-learn xgboost joblib openpyxl
2) uvicorn fuel_burn_api:app --host 0.0.0.0 --port 8600 --reload
"""

import os
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
import numpy as np


# =====================
# Konfigurasi dasar
# =====================

EXCEL_PATH = "05. Database RP May 2025 - AC REGISTER.xlsx"
SHEET_NAME = "Raw"

MODEL_PATH = "models/fuel_burn_xgb.joblib"

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

# Daftar fitur (sudah pakai nama yang di-rename)
SELECTED_FEATURES = [
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

CATEGORICAL_COLS = [
    "ROUNDTRIPROUTE",
    "AIRCRAFT_TYPE",
    "AC_REG",
    "SERVICE_TYPE",
    "FLIGHT_ROUTE",
]

NUMERIC_COLS = [c for c in SELECTED_FEATURES if c not in CATEGORICAL_COLS]
TARGET_COL = "FUEL_BURN_LITER"

# Global cache untuk model & encoder
_model_artifacts = None

# =====================
# Pydantic models
# =====================

class FuelBurnRecord(BaseModel):
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


class PredictRequest(BaseModel):
    records: List[FuelBurnRecord]


class PredictResponse(BaseModel):
    predictions: List[float]


class TrainResponse(BaseModel):
    mape: float          # dalam desimal, misal 0.05 = 5%
    mape_percent: float  # dalam persen
    rmse: float
    n_train: int
    n_test: int


# =====================
# Utility functions
# =====================

def load_training_data() -> pd.DataFrame:
    """Baca Excel, rename kolom, pilih fitur dan target, drop NaN."""
    if not os.path.exists(EXCEL_PATH):
        raise RuntimeError(f"File Excel tidak ditemukan: {EXCEL_PATH}")

    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, skiprows=1)
    df = df.iloc[:, 1:]  # buang kolom index pertama
    df = df[(df['FUEL BURN (IN LITER)']!=0) & (df['FLIGHT HOURS']!=0)].copy()

    df = df.rename(columns=RENAME_MAP)

    cols_needed = SELECTED_FEATURES + [TARGET_COL]
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"Kolom berikut tidak ditemukan di dataset: {missing}")

    df1 = df[cols_needed].dropna()
    
    return df1


def train_model():
    """Train XGBRegressor + OneHotEncoder, simpan artifacts, dan return metrics."""
    global _model_artifacts

    df1 = load_training_data()
    X = df1[SELECTED_FEATURES].copy()
    y = df1[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(X_train[CATEGORICAL_COLS])

    X_train_cat = encoder.transform(X_train[CATEGORICAL_COLS])
    X_test_cat = encoder.transform(X_test[CATEGORICAL_COLS])

    encoded_cols = encoder.get_feature_names_out(CATEGORICAL_COLS)

    X_train_cat_df = pd.DataFrame(
        X_train_cat, columns=encoded_cols, index=X_train.index
    )
    X_test_cat_df = pd.DataFrame(
        X_test_cat, columns=encoded_cols, index=X_test.index
    )

    X_train_final = pd.concat([X_train[NUMERIC_COLS], X_train_cat_df], axis=1)
    X_test_final = pd.concat([X_test[NUMERIC_COLS], X_test_cat_df], axis=1)

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
        "categorical_cols": CATEGORICAL_COLS,
        "numeric_cols": NUMERIC_COLS,
        "selected_features": SELECTED_FEATURES,
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(artifacts, MODEL_PATH)

    _model_artifacts = artifacts  # cache di memori

    return {
        "mape": float(mape),
        "mape_percent": float(mape * 100.0),
        "rmse": float(rmse),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }


def load_artifacts():
    """Load model & encoder dari disk jika belum ada di cache."""
    global _model_artifacts
    if _model_artifacts is not None:
        return _model_artifacts

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model belum dilatih. Jalankan endpoint /train dulu. "
            f"File tidak ditemukan: {MODEL_PATH}"
        )

    _model_artifacts = joblib.load(MODEL_PATH)
    return _model_artifacts


# =====================
# FastAPI app
# =====================

app = FastAPI(title="Fuel Burn XGBoost API", version="1.0")


@app.post("/train", response_model=TrainResponse)
def train_endpoint():
    """Latih ulang model dari file Excel."""
    try:
        metrics = train_model()
        return TrainResponse(**metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    """Prediksi fuel burn (dalam liter) dari fitur input."""
    try:
        artifacts = load_artifacts()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    model = artifacts["model"]
    encoder: OneHotEncoder = artifacts["encoder"]

    # Pydantic -> DataFrame
    data = pd.DataFrame([r.dict() for r in req.records])

    # Pastikan semua fitur ada
    missing = [c for c in SELECTED_FEATURES if c not in data.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Fitur berikut hilang di request: {missing}",
        )

    X_cat = data[CATEGORICAL_COLS]
    X_num = data[NUMERIC_COLS]

    X_cat_enc = encoder.transform(X_cat)
    encoded_cols = encoder.get_feature_names_out(CATEGORICAL_COLS)
    X_cat_enc_df = pd.DataFrame(X_cat_enc, columns=encoded_cols, index=data.index)

    X_final = pd.concat([X_num, X_cat_enc_df], axis=1)

    preds = model.predict(X_final)
    preds = [float(p) for p in preds]

    return PredictResponse(predictions=preds)


if __name__ == "__main__":

    import uvicorn

    uvicorn.run("fuel_burn_api:app", host="0.0.0.0", port=8600, reload=True)