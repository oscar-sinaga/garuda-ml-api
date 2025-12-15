"""
Docstring for passenger_commissions_api
Cara jalanin:
1) pip install fastapi uvicorn pandas scikit-learn xgboost joblib openpyxl
2) uvicorn passenger_commission_api:app --host 0.0.0.0 --port 8600 --reload

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
# from xgboost import XGBRegressor
import numpy as np
from lightgbm import LGBMRegressor

# =====================
# Konfigurasi dasar
# =====================

EXCEL_PATH = "05. Database RP May 2025 - AC REGISTER.xlsx"
SHEET_NAME = "Raw"

MODEL_PATH = "models/passenger_commission_lgbm.joblib"


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

SELECTED_FEATURES = [
    "RPK_000_C_CLASS",
    "RTK_000",
    "RPK_000",
    "RTK_PASSENGER_000",
    "RPK_000_Y_CLASS",
    "ASK_000_C_CLASS",
    "PASSENGER_CARRIED_C_CLASS"
]

TARGET_COL = "PASSENGER_COMMISSION"
_model_artifacts = None

# =====================
# Pydantic models
# =====================

class PassengerCommissionRecord(BaseModel):
    RPK_000_C_CLASS : float
    RTK_000 : float
    RPK_000 : float
    RTK_PASSENGER_000: float
    RPK_000_Y_CLASS: float
    ASK_000_C_CLASS: float
    PASSENGER_CARRIED_C_CLASS: float

class PredictRequest(BaseModel):
    records: List[PassengerCommissionRecord]


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
    if not os.path.exists(EXCEL_PATH):
        raise RuntimeError(f"File Excel tidak ditemukan: {EXCEL_PATH}")

    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, skiprows=1)
    df = df.iloc[:, 1:]  # buang kolom index pertama

    df = df.rename(columns=RENAME_MAP)

    ### REMOVE ZEROES
    df1 = df[(df['FLIGHT_KILOMETERS']!=0) & (df['SEAT_OFFERED']!=0) & 
            (df['PASSENGER_CARRIED']!=0) & (df['PASSENGER_COMMISSION']!=0) &
            (df['BLOCK_HOURS']!=0) & (df['FUEL_AIRCRAFT']>0)].copy()
    return df1

def train_model():
    """Train, simpan artifacts, dan return metrics."""
    global _model_artifacts

    df1 = load_training_data()
    X = df1[SELECTED_FEATURES].copy()
    y = df1[TARGET_COL].copy()

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

app = FastAPI(title="Passenger Commission API", version="1.0")

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

     # Pydantic -> DataFrame
    data = pd.DataFrame([r.dict() for r in req.records])

    # Pastikan semua fitur ada
    missing = [c for c in SELECTED_FEATURES if c not in data.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Fitur berikut hilang di request: {missing}",
        )

    preds = model.predict(data)
    preds = [float(p) for p in preds]

    return PredictResponse(predictions=preds)
