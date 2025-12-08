#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
API Variable Maintenance (XGBoost + OneHotEncoder)

Endpoint:
- POST /predict_vm -> prediksi VARIABLE MAINTENANCE
- POST /train_vm   -> latih ulang model VM dari Excel

Model disimpan di: models/vm_xgb.joblib
"""

import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

# =====================================================================
# KONFIGURASI
# =====================================================================

MODEL_PATH = "models/vm_xgb.joblib"

# FILE INPUT UNTUK TRAINING
VM_EXCEL_PATH = "05. Database RP May 2025 - AC REGISTER.xlsx"   
VM_SHEET = "Raw"                 


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


# =====================================================================
# GLOBAL CACHE
# =====================================================================

_vm_artifacts = None


# =====================================================================
# LOAD ARTIFACTS
# =====================================================================

def load_vm_artifacts():
    global _vm_artifacts
    if _vm_artifacts is not None:
        return _vm_artifacts

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model VM tidak ditemukan di {MODEL_PATH}")

    _vm_artifacts = joblib.load(MODEL_PATH)
    return _vm_artifacts


# =====================================================================
# TRAINING FUNCTION
# =====================================================================

def train_vm_model():
    """
    Melatih ulang model Variable Maintenance dari Excel.
    Menggunakan logika preprocessing persis seperti di notebook Oscar.
    """
    if not os.path.exists(VM_EXCEL_PATH):
        raise RuntimeError(f"File Excel VM tidak ditemukan: {VM_EXCEL_PATH}")

    # ============================ Load Data ============================
    df = pd.read_excel(VM_EXCEL_PATH, sheet_name=VM_SHEET, skiprows=1)
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


    # ========================== SELECT FEATURES ==========================
    selected_features = [
        "BLOCK_HOURS", "FLIGHT_HOURS", "FUEL_BURN_IN_LITER",
        "ASK_000", "ATK_PASSENGER_000", "ATK_000",
        "LEASE_AIRCRAFT", "CABIN_CREW_TRAVEL", "FUEL_AIRCRAFT",
        "COCKPIT_CREW_TRAVEL", "CABIN_CREW_PERSON",
        "ROUNDTRIPROUTE", "AIRCRAFT_TYPE", "SERVICE_TYPE",
        "AC_REG", "FLIGHT_ROUTE", "PERIODE",
        "FH_per_BH", "Fuel_per_FH", "ATK_per_ASK"
    ]


    categorical_cols = [
        "AC_REG", "PERIODE", "ROUNDTRIPROUTE",
        "AIRCRAFT_TYPE", "SERVICE_TYPE", "FLIGHT_ROUTE"
    ]


    numeric_cols = list(set(selected_features) - set(categorical_cols))

    X = df_group[selected_features].copy()
    y = df_group["VARIABLE MAINTENANCE"].copy()

    # =================== Split by AC REG (GroupSplit) ====================
    splitter = GroupShuffleSplit(test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(df_group, groups=df_group["AC_REG"]))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # =========================== ENCODER ================================
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(X_train[categorical_cols])

    X_train_enc = encoder.transform(X_train[categorical_cols])
    X_test_enc = encoder.transform(X_test[categorical_cols])

    enc_cols = encoder.get_feature_names_out(categorical_cols)

    X_train_enc_df = pd.DataFrame(X_train_enc, columns=enc_cols, index=X_train.index)
    X_test_enc_df = pd.DataFrame(X_test_enc, columns=enc_cols, index=X_test.index)

    X_train_final = pd.concat([X_train[numeric_cols], X_train_enc_df], axis=1)
    X_test_final = pd.concat([X_test[numeric_cols], X_test_enc_df], axis=1)

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
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "selected_features": selected_features
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump(artifacts, MODEL_PATH)

    return {
        "mape": float(mape),
        "mape_percent": float(mape * 100),
        "rmse": float(rmse),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test))
    }


# =====================================================================
# FASTAPI ENDPOINTS
# =====================================================================

app = FastAPI(title="Variable Maintenance API", version="2.0")


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


if __name__ == "__main__":

    import uvicorn

    uvicorn.run("vm_api:app", host="0.0.0.0", port=8700, reload=True)
