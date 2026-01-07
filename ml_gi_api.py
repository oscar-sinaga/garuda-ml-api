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
from fastapi.middleware.cors import CORSMiddleware
import traceback


# =====================================================================
# KONFIGURASI
# =====================================================================

# Sheet Excel 
# EXCEL_PATH = "05. Database RP May 2025 - AC REGISTER.xlsx"
EXCEL_PATH = "05. Database RP May 2025 - AC REGISTER.xlsx"
SHEET_NAME = "Raw"

# Model Path
VM_MODEL_PATH = "models/vm_xgb.joblib"
FB_MODEL_PATH = "models/fuel_burn_xgb.joblib"
PC_MODEL_PATH = "models/passenger_commission_lgbm.joblib"
FC_MODEL_PATH = "models/freight_commission_xgb.joblib"
R_MODEL_PATH = "models/reservation_xgb.joblib"
OBS_MODEL_PATH = "models/on_board_service_xgb.joblib"
C_MODEL_PATH = "models/catering_xgb.joblib"
MR_MODEL_PATH = "models/maintenance_reserve_xgb.joblib"
CF_COCKPIT_MODEL_PATH = "models/cockpit_xgb.joblib"
CF_CABIN_MODEL_PATH = "models/cabin_xgb.joblib"
STATION_MODEL_PATH = "models/station_xgb.joblib"
ADMIN_BO_MODEL_PATH = "models/admin_bo_xgb.joblib"
PAYROLL_COCKPIT_MODEL_PATH = "models/payroll_cockpit_xgb.joblib"
PAYROLL_CABIN_MODEL_PATH = "models/payroll_cabin_xgb.joblib"
LANDING_MODEL_PATH = "models/landing_xgb.joblib"
HANDLING_MODEL_PATH = "models/handling_xgb.joblib"
ATC_MODEL_PATH = "models/atc_xgb.joblib"





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

# FREIGHT COMMISSION FEATURES

SELECTED_FEATURES_FC = ['CLF_PERCENT', 
                        'CLF_GF_PERCENT', 
                        'LOAD_FACTOR_PERCENT', 
                        'CARGO_CARRIED', 
                        'FREIGHT_CARRIED',
                        'SERVICE_TYPE', 
                        'SUB_SERVICE', 
                        'FLIGHT_ROUTE'
]

TARGET_COL_FC = "FREIGHT_COMMISSION"

CATEGORICAL_COLS_FC = ['SERVICE_TYPE', 'SUB_SERVICE', 'FLIGHT_ROUTE']

NUMERICAL_COLS_FC = list(set(SELECTED_FEATURES_FC) - set(CATEGORICAL_COLS_FC))

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

# MR FEATURES
SELECTED_FEATURES_MR_MODEL = [
    'FLIGHT_HOURS',
    'FUEL_BURN_IN_LITER',
    'NUMBER_OF_LANDING',
    'ATK_000',
    'LEASE_AIRCRAFT',
    'AIRCRAFT_TYPE',
    'AC_REG',
    'PERIODE',
    'FH_per_Cycle' 
]

CATEGORICAL_COLS_MR = ['AC_REG', 'PERIODE', 'AIRCRAFT_TYPE']
NUMERICAL_COLS_MR = list(set(SELECTED_FEATURES_MR_MODEL) - set(CATEGORICAL_COLS_MR))
TARGET_COL_MR = "MAINTENANCE RESERVE"

# CREW FATA FEATURE
SELECTED_FEATURES_CREW_FATA = [
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
NUMERICAL_COLS_CREW = list(set(SELECTED_FEATURES_CREW_FATA) - set(CATEGORICAL_COLS_CREW))

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

# BRANCH OFFICE AND FIXED STATION COST

# ADMINISTRATION BRANCH OFFICE
SELECTED_FEATURES_ADMIN_BO = ['SALES_ORGANIZATION', 
                              'COCKPIT_CREW_TRAVEL', 
                              'ASK_000_C_CLASS', 
                              'ASK_000',
                              'CABIN_CREW_TRAVEL',
                              'PERIODE', 
                              'QUARTER', 
                              'REGION', 
                              'GA_SERVICE']

TARGET_COL_ADMIN_BO = "ADMINISTRATION_BO"

CATEGORICAL_COLS_ADMIN_BO = ['PERIODE', 
                              'QUARTER', 
                              'REGION', 
                              'GA_SERVICE']

NUMERICAL_COLS_ADMIN_BO = list(set(SELECTED_FEATURES_ADMIN_BO) - set(CATEGORICAL_COLS_ADMIN_BO))

# STATION
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

NUMERICAL_COLS_STATION = list(set(SELECTED_FEATURES_STATION) - set(CATEGORICAL_COLS_STATION))

# PAYROLL FEATURE
FEATURES_PAYROLL_COCKPIT = [
    'BLOCK_HOURS', 'FLIGHT_HOURS', 'FLIGHT_KILOMETERS', 
    'NUMBER_OF_LANDING', 'LEASE_AIRCRAFT', 
    'AIRCRAFT_TYPE', 'SERVICE_TYPE', 'PERIODE'
]

FEATURES_PAYROLL_CABIN = [
    'ASK_000_Y_CLASS', 'ASK_000_C_CLASS', 'BLOCK_HOURS',
    'FUEL_BURN_IN_LITER', 'LEASE_AIRCRAFT',
    'AIRCRAFT_TYPE', 'NUMBER_OF_LANDING', 'SERVICE_TYPE', 'PERIODE'
]

CATEGORICAL_COLS_PAYROLL = ['AIRCRAFT_TYPE', 'SERVICE_TYPE', 'PERIODE']

# AIRPORT FEES AND GROUND HANDLING FEATURE

## LANDING
NUMERICAL_COLS_LANDING = ['ATK_PASSENGER_000', 'ATK_000', 'BLOCK_HOURS']
CATEGORICAL_COLS_LANDING = ['AIRCRAFT_TYPE_GROUPING', 'FLIGHT_ROUTE', 'AC_REG', 'AIRCRAFT_TYPE']

SELECTED_FEATURES_LANDING = NUMERICAL_COLS_LANDING + CATEGORICAL_COLS_LANDING

## HANDLING
NUMERICAL_COLS_HANDLING = ['ATK_PASSENGER_000', 'ATK_000', 'BLOCK_HOURS']
CATEGORICAL_COLS_HANDLING = ['FLIGHT_ROUTE', 'AC_REG']
SELECTED_FEATURES_HANDLING = NUMERICAL_COLS_HANDLING + CATEGORICAL_COLS_HANDLING

## AIR TRAFFIC CONTROL
NUMERICAL_COLS_ATC = ['ATK_PASSENGER_000', 'ATK_000', 'BLOCK_HOURS']
CATEGORICAL_COLS_ATC = ['AIRCRAFT_TYPE_GROUPING', 'FLIGHT_ROUTE', 'ROUNDTRIPROUTE', 'AC_REG', 'AIRCRAFT_TYPE']
SELECTED_FEATURES_ATC = NUMERICAL_COLS_ATC + CATEGORICAL_COLS_ATC

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

# FREIGHT COMMISSION
class FCRecord(BaseModel):
    CLF_PERCENT : float
    CLF_GF_PERCENT : float
    LOAD_FACTOR_PERCENT : float
    CARGO_CARRIED: float
    FREIGHT_CARRIED: float
    SERVICE_TYPE: str
    SUB_SERVICE: str
    FLIGHT_ROUTE: str

class FCPredictRequest(BaseModel):
    records: List[FCRecord]

class FCPredictResponse(BaseModel):
    predictions: List[float]

class FCTrainResponse(BaseModel):
    mape: float          # dalam desimal, misal 0.05 = 5%
    mape_percent: float  # dalam persen
    rmse: float
    n_train: int
    n_test: int                

# Reservation
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


# CREW FATA
class CrewFATARecord(BaseModel):
    BLOCK_HOURS: float
    FLIGHT_KILOMETERS: float
    ASK_000: float
    NUMBER_OF_LANDING: float
    ATK_000: float
    SEAT_OFFERED: float
    AIRCRAFT_TYPE: str
    SERVICE_TYPE: str
    PERIODE: str

class CrewFATAPredictRequest(BaseModel):
    records: List[CrewFATARecord]

class CrewFATAPredictResponse(BaseModel):
    cockpit_predictions: List[float]
    cabin_predictions: List[float]

class SingleTrainMetric(BaseModel):
    mape: float
    mape_percent: float
    rmse: float
    n_train: int
    n_test: int

class CrewFATATrainResponse(BaseModel):
    cockpit: SingleTrainMetric
    cabin: SingleTrainMetric

# Branch office and fixed station cost
# Administration BO
class AdminBORecord(BaseModel):
    SALES_ORGANIZATION: float
    COCKPIT_CREW_TRAVEL: float
    ASK_000_C_CLASS: float
    ASK_000: float
    CABIN_CREW_TRAVEL: float
    PERIODE: str
    QUARTER: str
    REGION: str
    GA_SERVICE: str

class AdminBOPredictRequest(BaseModel):
    records: List[AdminBORecord]

class AdminBOPredictResponse(BaseModel):
    predictions: List[float]

class AdminBOTrainResponse(BaseModel):
    mape: float
    mape_percent: float
    rmse: float
    n_train: int
    n_test: int

# Station
class StationRecord(BaseModel):
    COCKPIT_CREW_TRAVEL: float
    FLIGHT_KILOMETERS: float
    COCKPIT_CREW_PERSON: float
    CABIN_CREW_TRAVEL:float
    PERIODE: str
    QUARTER: str
    REGION: str
    GA_SERVICE: str

class StationPredictRequest(BaseModel):
    records: List[StationRecord]

class StationPredictResponse(BaseModel):
    predictions: List[float]

class StationTrainResponse(BaseModel):
    mape: float
    mape_percent: float
    rmse: float
    n_train: int
    n_test: int

# Combined (Admin BO and Station)
class BOFSCPredictRequest(BaseModel):
    administration_bo: AdminBOPredictRequest
    station: StationPredictRequest

class BOFSCPredictResponse(BaseModel):
    administration_bo: AdminBOPredictResponse
    station: StationPredictResponse

class BOFSCTrainResponse(BaseModel):
    administration_bo : AdminBOTrainResponse
    station : StationTrainResponse


# PAYROLL


class PayrollRecord(BaseModel):
    AC_REG: str 
    PERIODE: str
    AIRCRAFT_TYPE: str
    SERVICE_TYPE: str
    
    BLOCK_HOURS: float
    FLIGHT_HOURS: float
    FLIGHT_KILOMETERS: float
    NUMBER_OF_LANDING: float
    LEASE_AIRCRAFT: float
    FUEL_BURN_IN_LITER: float
    ASK_000_Y_CLASS: float
    ASK_000_C_CLASS: float

class PayrollPredictRequest(BaseModel):
    records: List[PayrollRecord]

class PayrollPredictResponse(BaseModel):
    cockpit_person_cost: List[float]
    cabin_person_cost: List[float]

class PayrollTrainResponse(BaseModel):
    cockpit: SingleTrainMetric
    cabin: SingleTrainMetric


# Airport fees and ground handling

## LANDING
class LandingRecord(BaseModel):
    ATK_PASSENGER_000: float
    ATK_000: float
    BLOCK_HOURS: float
    AIRCRAFT_TYPE_GROUPING: str
    FLIGHT_ROUTE: str
    AC_REG: str
    AIRCRAFT_TYPE: str

class LandingPredictRequest(BaseModel):
    records: List[LandingRecord]

class LandingPredictResponse(BaseModel):
    predictions: List[float]

class LandingTrainResponse(BaseModel):
    mape: float
    mape_percent: float
    rmse: float
    n_train: int
    n_test: int

# HANDLING
class HandlingRecord(BaseModel):
    ATK_PASSENGER_000: float
    ATK_000: float
    BLOCK_HOURS: float
    FLIGHT_ROUTE: str
    AC_REG: str

class HandlingPredictRequest(BaseModel):
    records: List[HandlingRecord]

class HandlingPredictResponse(BaseModel):
    predictions: List[float]

class HandlingTrainResponse(BaseModel):
    mape: float
    mape_percent: float
    rmse: float
    n_train: int
    n_test: int

# ATC
class ATCRecord(BaseModel):
    ATK_PASSENGER_000: float
    ATK_000: float
    BLOCK_HOURS: float
    AIRCRAFT_TYPE_GROUPING: str
    FLIGHT_ROUTE: str
    ROUNDTRIPROUTE: str
    AC_REG: str
    AIRCRAFT_TYPE: str

class ATCPredictRequest(BaseModel):
    records: List[ATCRecord]

class ATCPredictResponse(BaseModel):
    predictions: List[float]

class ATCTrainResponse(BaseModel):
    mape: float
    mape_percent: float
    rmse: float
    n_train: int
    n_test: int


# Combined (handing, landing, atc)
class AFGHPredictRequest(BaseModel):
    landing: LandingPredictRequest
    handling: HandlingPredictRequest
    atc: ATCPredictRequest

class AFGHPredictResponse(BaseModel):
    landing: LandingPredictResponse
    handling: HandlingPredictResponse
    atc: ATCPredictResponse

class AFGHTrainResponse(BaseModel):
    landing: LandingTrainResponse
    handling: HandlingTrainResponse
    atc: ATCTrainResponse   

# =====================================================================
# GLOBAL CACHE
# =====================================================================

_vm_artifacts = None

# Global cache untuk model & encoder
_fb_artifacts = None

_pc_model_artifacts = None

_fc_model_artifacts = None

_reservation_model_artifacts = None

_obs_model_artifacts = None
_catering_model_artifacts = None

_mr_artifacts = None

_cockpit_artifacts = None
_cabin_artifacts = None

_admin_bo_model_artifacts = None
_station_model_artifacts = None

_payroll_cockpit_artifacts = None
_payroll_cabin_artifacts = None

_landing_model_artifacts = None
_handling_model_artifacts = None
_atc_model_artifacts = None

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

def load_fc_artifacts():
    """Load model & encoder dari disk jika belum ada di cache."""
    global _fc_model_artifacts
    if _fc_model_artifacts is not None:
        return _fc_model_artifacts

    if not os.path.exists(FC_MODEL_PATH):
        raise RuntimeError(
            f"Model belum dilatih. Jalankan endpoint /train dulu. "
            f"File tidak ditemukan: {FC_MODEL_PATH}"
        )

    _fc_model_artifacts = joblib.load(FC_MODEL_PATH)
    return _fc_model_artifacts

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

#=====================
# CREW FATA
#=====================
def load_crew_artifacts(target_type="cockpit"):
    """Load model specific to target (cockpit/cabin)"""
    global _cockpit_artifacts, _cabin_artifacts
    
    path = CF_COCKPIT_MODEL_PATH if target_type == "cockpit" else CF_CABIN_MODEL_PATH
    
    if target_type == "cockpit":
        if _cockpit_artifacts is not None: return _cockpit_artifacts
    else:
        if _cabin_artifacts is not None: return _cabin_artifacts

    if not os.path.exists(path):
        raise RuntimeError(f"Model {target_type} belum dilatih. File tidak ditemukan: {path}")

    artifacts = joblib.load(path)
    
    if target_type == "cockpit":
        _cockpit_artifacts = artifacts
        return _cockpit_artifacts
    else:
        _cabin_artifacts = artifacts
        return _cabin_artifacts


#=========================

def load_admin_bo_artifacts():
    """Load model & encoder dari disk jika belum ada di cache."""
    global _admin_bo_model_artifacts
    if _admin_bo_model_artifacts is not None:
        return _admin_bo_model_artifacts

    if not os.path.exists(ADMIN_BO_MODEL_PATH):
        raise RuntimeError(
            f"Model belum dilatih. Jalankan endpoint /train dulu. "
            f"File tidak ditemukan: {ADMIN_BO_MODEL_PATH}"
        )

    _admin_bo_model_artifacts = joblib.load(ADMIN_BO_MODEL_PATH)
    return _admin_bo_model_artifacts

def load_station_artifacts():
    """Load model & encoder dari disk jika belum ada di cache."""
    global _station_model_artifacts
    if _station_model_artifacts is not None:
        return _station_model_artifacts

    if not os.path.exists(STATION_MODEL_PATH):
        raise RuntimeError(
            f"Model belum dilatih. Jalankan endpoint /train dulu. "
            f"File tidak ditemukan: {STATION_MODEL_PATH}"
        )

    _station_model_artifacts = joblib.load(STATION_MODEL_PATH)
    return _station_model_artifacts



# ================
# Payroll

def load_payroll_artifacts(role="cockpit"):
    global _payroll_cockpit_artifacts, _payroll_cabin_artifacts
    path = PAYROLL_COCKPIT_MODEL_PATH if role == "cockpit" else PAYROLL_CABIN_MODEL_PATH
    
    # Check memory cache
    if role == "cockpit" and _payroll_cockpit_artifacts: return _payroll_cockpit_artifacts
    if role == "cabin" and _payroll_cabin_artifacts: return _payroll_cabin_artifacts
    
    if not os.path.exists(path):
        raise RuntimeError(f"Model Payroll {role} belum dilatih.")
        
    artifacts = joblib.load(path)
    
    if role == "cockpit": _payroll_cockpit_artifacts = artifacts
    else: _payroll_cabin_artifacts = artifacts
    
    return artifacts

#========================
def load_landing_artifacts():
    global _landing_model_artifacts
    if _landing_model_artifacts is not None:
        return _landing_model_artifacts

    if not os.path.exists(LANDING_MODEL_PATH):
        raise RuntimeError(f"Model Landing belum dilatih. Jalankan endpoint /train_landing dulu.")

    _landing_model_artifacts = joblib.load(LANDING_MODEL_PATH)
    return _landing_model_artifacts

def load_handling_artifacts():
    global _handling_model_artifacts
    if _handling_model_artifacts is not None:
        return _handling_model_artifacts

    if not os.path.exists(HANDLING_MODEL_PATH):
        raise RuntimeError(f"Model Handling belum dilatih. Jalankan endpoint /train_handling dulu.")

    _handling_model_artifacts = joblib.load(HANDLING_MODEL_PATH)
    return _handling_model_artifacts

def load_atc_artifacts():
    global _atc_model_artifacts
    if _atc_model_artifacts is not None:
        return _atc_model_artifacts

    if not os.path.exists(ATC_MODEL_PATH):
        raise RuntimeError(f"Model ATC belum dilatih. Jalankan endpoint /train_atc dulu.")

    _atc_model_artifacts = joblib.load(ATC_MODEL_PATH)
    return _atc_model_artifacts

#========================

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

def train_fc_model():
    """Train, simpan artifacts, dan return metrics."""
    global _fc_model_artifacts

    def load_training_data() -> pd.DataFrame:
        if not os.path.exists(EXCEL_PATH):
            raise RuntimeError(f"File Excel tidak ditemukan: {EXCEL_PATH}")

        df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, skiprows=1)
        df = df.iloc[:, 1:]  # buang kolom index pertama

        # REMOVE ZEROES
        df1 = df[
        (df['FREIGHT COMMISSION']>=10) & (df['FREIGHT CARRIED']>0) & (df['CARGO CARRIED']>0) &
        (df['FREIGHT COMMISSION']<=1000) &
        (df['BLOCK HOURS']>0) & (df['FLIGHT HOURS']>0) & (df['ASK (000)']>0)].copy()

        # Mapping nama kolom Excel -> nama Pythonic
        RENAME_MAP = {
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

        df1 = df1.rename(columns=RENAME_MAP)
        return df1
    
    df1 = load_training_data()
    
    X = df1[SELECTED_FEATURES_FC].copy()
    y = df1[TARGET_COL_FC].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )
    

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(X_train[CATEGORICAL_COLS_FC])

    X_train_encoded = encoder.transform(X_train[CATEGORICAL_COLS_FC])
    X_test_encoded = encoder.transform(X_test[CATEGORICAL_COLS_FC])

    encoded_cols = encoder.get_feature_names_out(CATEGORICAL_COLS_FC)
    
    X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoded_cols, index=X_train.index)
    X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoded_cols, index=X_test.index)
    
    X_train_final = pd.concat([X_train[NUMERICAL_COLS_FC], X_train_encoded], axis=1)
    X_test_final = pd.concat([X_test[NUMERICAL_COLS_FC], X_test_encoded], axis=1)

    model = XGBRegressor(n_estimators=1000, 
                         learning_rate=0.5, 
                         objective="reg:squarederror")
    
    model.fit(X_train_final, y_train)
    
    y_pred = model.predict(X_test_final)

    mape = mean_absolute_percentage_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    artifacts = {
        "model": model,
        "encoder": encoder,
        "categorical_cols": CATEGORICAL_COLS_FC,
        "numeric_cols": NUMERICAL_COLS_FC,
        "selected_features": SELECTED_FEATURES_FC,
    }

    os.makedirs(os.path.dirname(FC_MODEL_PATH), exist_ok=True)
     
    joblib.dump(artifacts, FC_MODEL_PATH)
     
    _fc_model_artifacts = artifacts  # cache di memori
    
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


#=============================================
# MAINTENANCE RESERVE
#=============================================

def train_mr_model():
    global _mr_artifacts
    
    if not os.path.exists(EXCEL_PATH):
        raise RuntimeError(f"File Excel tidak ditemukan: {EXCEL_PATH}")

    # 1. Load Data
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, skiprows=1)
    df = df.iloc[:, 1:] 
    
    # 2. Filter Data 
    if 'MAINTENANCE RESERVE' not in df.columns:
         raise RuntimeError("Kolom 'MAINTENANCE RESERVE' tidak ada di Excel.")

    df = df[df['MAINTENANCE RESERVE'] >= 0].copy()
    
    df1 = df[
        (df['FUEL BURN (IN LITER)'] != 0) &
        (df['FLIGHT HOURS'] != 0) &
        (df['MAINTENANCE RESERVE'] != 0)
    ].copy()

    # 3. Grouping 
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
    train_idx, test_idx = next(splitter.split(df_group, groups=df_group["AC_REG"]))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # 7. Encoding
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(X_train[CATEGORICAL_COLS_MR])
    
    X_train_enc = encoder.transform(X_train[CATEGORICAL_COLS_MR])
    X_test_enc = encoder.transform(X_test[CATEGORICAL_COLS_MR])
    
    encoded_cols = encoder.get_feature_names_out(CATEGORICAL_COLS_MR)
    
    #Numeric + Encoded
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

#=============================================
# Crew FATA
#=============================================

def train_single_crew_model(target_col, model_path):
    """Fungsi generik untuk train cockpit atau cabin"""
    if not os.path.exists(EXCEL_PATH):
        raise RuntimeError(f"File Excel tidak ditemukan: {EXCEL_PATH}")

    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, skiprows=1)
    df = df.iloc[:, 1:]

    # Rename Mapping
    rename_map = {
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
    df.rename(columns=rename_map, inplace=True)

    # Filter Logic
    df_clean = df[
        (df['BLOCK_HOURS'] > 0) & 
        (df['ASK_000'] > 0) &
        (df[target_col] >= 0) 
    ].copy()

    X = df_clean[SELECTED_FEATURES_CREW_FATA].copy()
    y = df_clean[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(X_train[CATEGORICAL_COLS_CREW])

    X_train_enc = encoder.transform(X_train[CATEGORICAL_COLS_CREW])
    X_test_enc = encoder.transform(X_test[CATEGORICAL_COLS_CREW])
    enc_cols = encoder.get_feature_names_out(CATEGORICAL_COLS_CREW)

    X_train_final = pd.concat([
        X_train[NUMERICAL_COLS_CREW].reset_index(drop=True),
        pd.DataFrame(X_train_enc, columns=enc_cols, index=X_train.index).reset_index(drop=True)
    ], axis=1)

    X_test_final = pd.concat([
        X_test[NUMERICAL_COLS_CREW].reset_index(drop=True),
        pd.DataFrame(X_test_enc, columns=enc_cols, index=X_test.index).reset_index(drop=True)
    ], axis=1)

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        objective="reg:squarederror"
    )
    model.fit(X_train_final, y_train)

    y_pred = model.predict(X_test_final)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    artifacts = {
        "model": model,
        "encoder": encoder,
        "categorical_cols": CATEGORICAL_COLS_CREW,
        "numeric_cols": NUMERICAL_COLS_CREW,
        "selected_features": SELECTED_FEATURES_CREW_FATA
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(artifacts, model_path)

    return {
        "mape": float(mape),
        "mape_percent": float(mape * 100),
        "rmse": float(rmse),
        "n_train": len(X_train),
        "n_test": len(X_test)
    }


# ADMINISTRATION BO AND STATION
def train_admin_bo_model():
    """Train, simpan artifacts, dan return metrics."""
    global _admin_bo_model_artifacts
     
    def load_training_data() -> pd.DataFrame:
        if not os.path.exists(EXCEL_PATH):
            raise RuntimeError(f"File Excel tidak ditemukan: {EXCEL_PATH}")

        df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, skiprows=1)
        df = df.iloc[:, 1:]  # buang kolom index pertama

        df1 = df[(df['STATION']>1) & (df['ADMINISTRATION BO']>1) & 
                (df['BLOCK HOURS']>0) & (df['FLIGHT HOURS']>0) & 
                (df['ASK (000)']>0)].copy()
        
        RENAME_MAP = {
            'SALES ORGANIZATION': 'SALES_ORGANIZATION', 
            'COCKPIT CREW TRAVEL': 'COCKPIT_CREW_TRAVEL', 
            'ASK (000) C CLASS': 'ASK_000_C_CLASS', 
            'ASK (000)': 'ASK_000',
            'CABIN CREW TRAVEL': 'CABIN_CREW_TRAVEL',
            'Region': 'REGION', 
            'GA Service': 'GA_SERVICE',
            'ADMINISTRATION BO': 'ADMINISTRATION_BO'
        }

        df1 = df1.rename(columns=RENAME_MAP)
        return df1
    
    df1 = load_training_data()
    X = df1[SELECTED_FEATURES_ADMIN_BO].copy()
    y = df1['ADMINISTRATION_BO'].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=25)
    
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(X_train[CATEGORICAL_COLS_ADMIN_BO])

    X_train_encoded = encoder.transform(X_train[CATEGORICAL_COLS_ADMIN_BO])
    X_test_encoded = encoder.transform(X_test[CATEGORICAL_COLS_ADMIN_BO])

    encoded_cols = encoder.get_feature_names_out(CATEGORICAL_COLS_ADMIN_BO)

    X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoded_cols, index=X_train.index)
    X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoded_cols, index=X_test.index)

    X_train_final = pd.concat([X_train[NUMERICAL_COLS_ADMIN_BO], X_train_encoded], axis=1)
    X_test_final = pd.concat([X_test[NUMERICAL_COLS_ADMIN_BO], X_test_encoded], axis=1)

    model = XGBRegressor(n_estimators=2000, learning_rate=0.5, objective="reg:squarederror")
    model.fit(X_train_final, y_train)

    y_pred = model.predict(X_test_final)

    mape = mean_absolute_percentage_error(y_test, y_pred)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    artifacts = {
        "model": model,
        "encoder": encoder,
        "categorical_cols": CATEGORICAL_COLS_ADMIN_BO,
        "numeric_cols": NUMERICAL_COLS_ADMIN_BO,
        "selected_features": SELECTED_FEATURES_ADMIN_BO,
    }

    os.makedirs(os.path.dirname(ADMIN_BO_MODEL_PATH), exist_ok=True)
    
    joblib.dump(artifacts, ADMIN_BO_MODEL_PATH)
    
    _admin_bo_model_artifacts = artifacts  # cache di memori
    
    return {
        "mape": float(mape),
        "mape_percent": float(mape * 100.0),
        "rmse": float(rmse),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

def train_station_model():
    """Train, simpan artifacts, dan return metrics."""
    global _admin_bo_model_artifacts

    def load_training_data() -> pd.DataFrame:
        if not os.path.exists(EXCEL_PATH):
            raise RuntimeError(f"File Excel tidak ditemukan: {EXCEL_PATH}")

        df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, skiprows=1)
        df = df.iloc[:, 1:]  # buang kolom index pertama

        df1 = df[(df['STATION']>1) & (df['ADMINISTRATION BO']>1) & 
                (df['FLIGHT KILOMETERS']>0) & 
                (df['ASK (000)']>0)].copy()
        
        RENAME_MAP = {
            'COCKPIT CREW TRAVEL': 'COCKPIT_CREW_TRAVEL', 
            'CABIN CREW TRAVEL': 'CABIN_CREW_TRAVEL',
            'FLIGHT KILOMETERS': 'FLIGHT_KILOMETERS', 
            'COCKPIT CREW PERSON': 'COCKPIT_CREW_PERSON',
            'Region': 'REGION', 
            'GA Service': 'GA_SERVICE',
            'ADMINISTRATION BO': 'ADMINISTRATION_BO'
        }

        df1 = df1.rename(columns=RENAME_MAP)
        return df1
    
    df1 = load_training_data()
    X = df1[SELECTED_FEATURES_STATION].copy()
    y = df1['STATION'].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=10)
    
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(X_train[CATEGORICAL_COLS_STATION])

    X_train_encoded = encoder.transform(X_train[CATEGORICAL_COLS_STATION])
    X_test_encoded = encoder.transform(X_test[CATEGORICAL_COLS_STATION])

    encoded_cols = encoder.get_feature_names_out(CATEGORICAL_COLS_STATION)

    X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoded_cols, index=X_train.index)
    X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoded_cols, index=X_test.index)

    X_train_final = pd.concat([X_train[NUMERICAL_COLS_STATION], X_train_encoded], axis=1)
    X_test_final = pd.concat([X_test[NUMERICAL_COLS_STATION], X_test_encoded], axis=1)

    model = XGBRegressor(n_estimators=1500, learning_rate=0.05, objective="reg:squarederror")
    model.fit(X_train_final, y_train)

    y_pred = model.predict(X_test_final)

    mape = mean_absolute_percentage_error(y_test, y_pred)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    artifacts = {
        "model": model,
        "encoder": encoder,
        "categorical_cols": CATEGORICAL_COLS_STATION,
        "numeric_cols": NUMERICAL_COLS_STATION,
        "selected_features": SELECTED_FEATURES_STATION,
    }

    os.makedirs(os.path.dirname(STATION_MODEL_PATH), exist_ok=True)
    
    joblib.dump(artifacts, STATION_MODEL_PATH)
    
    _station_model_artifacts = artifacts  # cache di memori
    
    return {
        "mape": float(mape),
        "mape_percent": float(mape * 100.0),
        "rmse": float(rmse),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    
#=============================================
# PAYROLL
#=============================================
def train_single_payroll_model(target_col, feature_list, model_path):
    if not os.path.exists(EXCEL_PATH): raise RuntimeError("Excel not found")

    # 1. Load & Rename
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, skiprows=1)
    df = df.iloc[:, 1:]
    
    rename_map = {
        'BLOCK HOURS': 'BLOCK_HOURS', 'FLIGHT HOURS': 'FLIGHT_HOURS',
        'FLIGHT KILOMETERS': 'FLIGHT_KILOMETERS', 'NUMBER OF LANDING': 'NUMBER_OF_LANDING',
        'LEASE AIRCRAFT': 'LEASE_AIRCRAFT', 'AIRCRAFT TYPE': 'AIRCRAFT_TYPE',
        'SERVICE TYPE': 'SERVICE_TYPE', 'PERIODE': 'PERIODE', 'AC REG': 'AC_REG',
        'ASK (000) Y CLASS': 'ASK_000_Y_CLASS', 'ASK (000) C CLASS': 'ASK_000_C_CLASS',
        'FUEL BURN (IN LITER)': 'FUEL_BURN_IN_LITER',
        'COCKPIT CREW PERSON': 'COCKPIT_CREW_PERSON', 'CABIN CREW PERSON': 'CABIN_CREW_PERSON',
        'ASK (000)': 'ASK_000'
    }
    df.rename(columns=rename_map, inplace=True)
    
    # 2. Filter & Grouping (Sesuai Notebook)
    df1 = df[(df['BLOCK_HOURS'] > 0) & (df['ASK_000'] > 0)].copy()
    
    # Aggregation rules
    agg_rules = {
        target_col: "sum",
        "BLOCK_HOURS": "sum", "FLIGHT_HOURS": "sum", "FLIGHT_KILOMETERS": "sum",
        "ASK_000_Y_CLASS": "sum", "ASK_000_C_CLASS": "sum",
        "LEASE_AIRCRAFT": "mean", "FUEL_BURN_IN_LITER": "sum", "NUMBER_OF_LANDING": "sum",
        "AIRCRAFT_TYPE": "first", "SERVICE_TYPE": "first"
    }
    
    df_group = df1.groupby(["AC_REG", "PERIODE"]).agg(agg_rules).reset_index()
    
    # 3. Filter Target > 0
    data_train = df_group[df_group[target_col] > 0].copy()
    
    # 4. Split GroupShuffleSplit by AC_REG
    X = data_train[feature_list + ['AC_REG']] 
    y = data_train[target_col]
    
    splitter = GroupShuffleSplit(test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=X["AC_REG"]))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # 5. LOG TRANSFORMATION (Important!)
    y_train_log = np.log1p(y_train)
    
    # 6. Encoding
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(X_train[CATEGORICAL_COLS_PAYROLL])
    
    X_train_enc = pd.DataFrame(encoder.transform(X_train[CATEGORICAL_COLS_PAYROLL]), 
                               columns=encoder.get_feature_names_out(CATEGORICAL_COLS_PAYROLL), 
                               index=X_train.index)
    X_test_enc = pd.DataFrame(encoder.transform(X_test[CATEGORICAL_COLS_PAYROLL]), 
                              columns=encoder.get_feature_names_out(CATEGORICAL_COLS_PAYROLL), 
                              index=X_test.index)
    
    # Numeric cols 
    num_cols = list(set(feature_list) - set(CATEGORICAL_COLS_PAYROLL))
    
    X_train_final = pd.concat([X_train[num_cols], X_train_enc], axis=1)
    X_test_final = pd.concat([X_test[num_cols], X_test_enc], axis=1)
    
    # 7. Train XGBoost
    model = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.01, 
                         subsample=0.8, colsample_bytree=0.8, n_jobs=-1, objective="reg:squarederror")
    model.fit(X_train_final, y_train_log)
    
    # 8. Evaluate (Inverse Log)
    y_pred_log = model.predict(X_test_final)
    y_pred_real = np.expm1(y_pred_log)
    
    mape = mean_absolute_percentage_error(y_test, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_real))
    
    # 9. Save Artifacts
    artifacts = {
        "model": model, "encoder": encoder,
        "features": feature_list, "categorical": CATEGORICAL_COLS_PAYROLL,
        "numeric": num_cols
    }
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(artifacts, model_path)
    
    return {
        "mape": float(mape), "mape_percent": float(mape*100),
        "rmse": float(rmse), "n_train": len(X_train), "n_test": len(X_test)
    }


# Airport Fees and ground handling 

## Landing
def train_landing_model():
    """Train, simpan artifacts, dan return metrics."""
    global _landing_model_artifacts
     
    def load_training_data() -> pd.DataFrame:
        if not os.path.exists(EXCEL_PATH):
            raise RuntimeError(f"File Excel tidak ditemukan: {EXCEL_PATH}")

        df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, skiprows=1)
        df = df.iloc[:, 1:]  # buang kolom index pertama

        df1 = df[
                (df['BLOCK HOURS'] > 0) & 
                (df['ASK (000)'] > 0) &
                (df['LANDING'] > 0) &
                (df['HANDLING'] > 0) &
                (df['AIR TRAFFIC CONTROL'] > 0)
            ].copy()
        
        RENAME_MAP = {
            'ATK (000)': 'ATK_000',
            'ATK PASSENGER (000)': 'ATK_PASSENGER_000',
            'BLOCK HOURS': 'BLOCK_HOURS',
            'AIRCRAFT TYPE GROUPING': 'AIRCRAFT_TYPE_GROUPING',
            'FLIGHT ROUTE': 'FLIGHT_ROUTE',
            'AC REG': 'AC_REG',
            'AIRCRAFT TYPE': 'AIRCRAFT_TYPE'
        }

        df1 = df1.rename(columns=RENAME_MAP)
        return df1
    
    df1 = load_training_data()
    X = df1[SELECTED_FEATURES_LANDING].copy()
    y = df1['LANDING'].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=25)
    
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(X_train[CATEGORICAL_COLS_LANDING])

    X_train_encoded = encoder.transform(X_train[CATEGORICAL_COLS_LANDING])
    X_test_encoded = encoder.transform(X_test[CATEGORICAL_COLS_LANDING])

    encoded_cols = encoder.get_feature_names_out(CATEGORICAL_COLS_LANDING)

    X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoded_cols, index=X_train.index)
    X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoded_cols, index=X_test.index)

    X_train_final = pd.concat([X_train[NUMERICAL_COLS_LANDING], X_train_encoded], axis=1)
    X_test_final = pd.concat([X_test[NUMERICAL_COLS_LANDING], X_test_encoded], axis=1)

    model = XGBRegressor(n_estimators=2000, max_depth=6, learning_rate=0.05, n_jobs=-1, objective="reg:squarederror")
    model.fit(X_train_final, y_train)

    y_pred = model.predict(X_test_final)

    mape = mean_absolute_percentage_error(y_test, y_pred)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    artifacts = {
        "model": model,
        "encoder": encoder,
        "categorical_cols": CATEGORICAL_COLS_LANDING,
        "numeric_cols": NUMERICAL_COLS_LANDING,
        "selected_features": SELECTED_FEATURES_LANDING,
    }

    os.makedirs(os.path.dirname(LANDING_MODEL_PATH), exist_ok=True)
    
    joblib.dump(artifacts, LANDING_MODEL_PATH)
    
    _landing_model_artifacts = artifacts  # cache di memori
    
    return {
        "mape": float(mape),
        "mape_percent": float(mape * 100.0),
        "rmse": float(rmse),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

# HANDLING
def train_handling_model():
    """Train, simpan artifacts, dan return metrics."""
    global _handling_model_artifacts
     
    def load_training_data() -> pd.DataFrame:
        if not os.path.exists(EXCEL_PATH):
            raise RuntimeError(f"File Excel tidak ditemukan: {EXCEL_PATH}")

        df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, skiprows=1)
        df = df.iloc[:, 1:]  # buang kolom index pertama

        df1 = df[
                (df['BLOCK HOURS'] > 0) & 
                (df['ASK (000)'] > 0) &
                (df['LANDING'] > 0) &
                (df['HANDLING'] > 0) &
                (df['AIR TRAFFIC CONTROL'] > 0)
            ].copy()
        
        RENAME_MAP = {
            'ATK (000)': 'ATK_000',
            'ATK PASSENGER (000)': 'ATK_PASSENGER_000',
            'BLOCK HOURS': 'BLOCK_HOURS',
            'FLIGHT ROUTE': 'FLIGHT_ROUTE',
            'AC REG': 'AC_REG'
        }

        df1 = df1.rename(columns=RENAME_MAP)
        return df1
    
    df1 = load_training_data()
    X = df1[SELECTED_FEATURES_HANDLING].copy()
    y = df1['HANDLING'].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=25)
    
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(X_train[CATEGORICAL_COLS_HANDLING])

    X_train_encoded = encoder.transform(X_train[CATEGORICAL_COLS_HANDLING])
    X_test_encoded = encoder.transform(X_test[CATEGORICAL_COLS_HANDLING])

    encoded_cols = encoder.get_feature_names_out(CATEGORICAL_COLS_HANDLING)

    X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoded_cols, index=X_train.index)
    X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoded_cols, index=X_test.index)

    X_train_final = pd.concat([X_train[NUMERICAL_COLS_HANDLING], X_train_encoded], axis=1)
    X_test_final = pd.concat([X_test[NUMERICAL_COLS_HANDLING], X_test_encoded], axis=1)

    model = XGBRegressor(n_estimators=2000, max_depth=6, learning_rate=0.05, n_jobs=-1, objective="reg:squarederror")
    model.fit(X_train_final, y_train)

    y_pred = model.predict(X_test_final)

    mape = mean_absolute_percentage_error(y_test, y_pred)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    artifacts = {
        "model": model,
        "encoder": encoder,
        "categorical_cols": CATEGORICAL_COLS_HANDLING,
        "numeric_cols": NUMERICAL_COLS_HANDLING,
        "selected_features": SELECTED_FEATURES_HANDLING,
    }

    os.makedirs(os.path.dirname(HANDLING_MODEL_PATH), exist_ok=True)
    
    joblib.dump(artifacts, HANDLING_MODEL_PATH)
    
    _handling_model_artifacts = artifacts  # cache di memori
    
    return {
        "mape": float(mape),
        "mape_percent": float(mape * 100.0),
        "rmse": float(rmse),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }


# ATC
def train_atc_model():
    """Train, simpan artifacts, dan return metrics."""
    global _atc_model_artifacts
     
    def load_training_data() -> pd.DataFrame:
        if not os.path.exists(EXCEL_PATH):
            raise RuntimeError(f"File Excel tidak ditemukan: {EXCEL_PATH}")

        df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, skiprows=1)
        df = df.iloc[:, 1:]  # buang kolom index pertama

        df1 = df[
                (df['BLOCK HOURS'] > 0) & 
                (df['ASK (000)'] > 0) &
                (df['LANDING'] > 0) &
                (df['HANDLING'] > 0) &
                (df['AIR TRAFFIC CONTROL'] > 0)
            ].copy()
        
        RENAME_MAP = {
            'ATK (000)': 'ATK_000',
            'ATK PASSENGER (000)': 'ATK_PASSENGER_000',
            'BLOCK HOURS': 'BLOCK_HOURS',
            'AIRCRAFT TYPE GROUPING': 'AIRCRAFT_TYPE_GROUPING',
            'FLIGHT ROUTE': 'FLIGHT_ROUTE',
            'AC REG': 'AC_REG',
            'AIRCRAFT TYPE': 'AIRCRAFT_TYPE'
        }

        df1 = df1.rename(columns=RENAME_MAP)
        return df1
    
    df1 = load_training_data()
    X = df1[SELECTED_FEATURES_ATC].copy()
    y = df1['HANDLING'].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=25)
    
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(X_train[CATEGORICAL_COLS_ATC])

    X_train_encoded = encoder.transform(X_train[CATEGORICAL_COLS_ATC])
    X_test_encoded = encoder.transform(X_test[CATEGORICAL_COLS_ATC])

    encoded_cols = encoder.get_feature_names_out(CATEGORICAL_COLS_ATC)

    X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoded_cols, index=X_train.index)
    X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoded_cols, index=X_test.index)

    X_train_final = pd.concat([X_train[NUMERICAL_COLS_ATC], X_train_encoded], axis=1)
    X_test_final = pd.concat([X_test[NUMERICAL_COLS_ATC], X_test_encoded], axis=1)

    model = XGBRegressor(n_estimators=2000, max_depth=6, learning_rate=0.05, n_jobs=-1, objective="reg:squarederror")
    model.fit(X_train_final, y_train)

    y_pred = model.predict(X_test_final)

    mape = mean_absolute_percentage_error(y_test, y_pred)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    artifacts = {
        "model": model,
        "encoder": encoder,
        "categorical_cols": CATEGORICAL_COLS_ATC,
        "numeric_cols": NUMERICAL_COLS_ATC,
        "selected_features": SELECTED_FEATURES_ATC,
    }

    os.makedirs(os.path.dirname(ATC_MODEL_PATH), exist_ok=True)
    
    joblib.dump(artifacts, ATC_MODEL_PATH)
    
    _atc_model_artifacts = artifacts  # cache di memori
    
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

@app.post("/predict_fc", response_model=FCPredictResponse)
def predict_fc(req: FCPredictRequest):
    try:
        artifacts = load_fc_artifacts()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    model = artifacts["model"]
    encoder = artifacts["encoder"]
    categorical_cols = artifacts["categorical_cols"]
    numeric_cols = artifacts["numeric_cols"]

     # Pydantic -> DataFrame
    df = pd.DataFrame([r.dict() for r in req.records])

    # Pastikan semua fitur ada
    missing = [c for c in SELECTED_FEATURES_FC if c not in df.columns]
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

    return FCPredictResponse(predictions= preds)


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

@app.post("/predict_crew_fata", response_model=CrewFATAPredictResponse)
def predict_crew_fata(req: CrewFATAPredictRequest):
    try:
        # Load artifacts
        art_cockpit = load_crew_artifacts("cockpit")
        art_cabin = load_crew_artifacts("cabin")

        # Prepare Data
        df = pd.DataFrame([r.dict() for r in req.records])

        # Preprocessing (Sama untuk kedua model karena fiturnya sama)
        encoder = art_cockpit["encoder"] # Encoder bisa pakai salah satu asalkan fiturnya sama persis
        cat_cols = art_cockpit["categorical_cols"]
        num_cols = art_cockpit["numeric_cols"]

        df_cat = df[cat_cols]
        df_num = df[num_cols]

        df_cat_enc = encoder.transform(df_cat)
        enc_cols = encoder.get_feature_names_out(cat_cols)

        X_final = pd.concat([
            df_num.reset_index(drop=True),
            pd.DataFrame(df_cat_enc, columns=enc_cols)
        ], axis=1)

        # Predict
        pred_cockpit = art_cockpit["model"].predict(X_final)
        pred_cabin = art_cabin["model"].predict(X_final)

        return {
            "cockpit_predictions": [float(p) for p in pred_cockpit],
            "cabin_predictions": [float(p) for p in pred_cabin]
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/predict_bofsc", response_model=BOFSCPredictResponse)
def predict_bofsc(req: BOFSCPredictRequest):
    try:
        artifacts_admin_bo = load_admin_bo_artifacts()
        artifacts_station = load_station_artifacts()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    model_admin_bo = artifacts_admin_bo["model"]
    encoder_admin_bo = artifacts_admin_bo["encoder"]
    categorical_admin_bo = artifacts_admin_bo["categorical_cols"]
    numeric_cols_admin_bo = artifacts_admin_bo["numeric_cols"]

    model_station = artifacts_station["model"]
    encoder_station = artifacts_station["encoder"]
    categorical_cols_station = artifacts_station["categorical_cols"]
    numeric_cols_station = artifacts_station["numeric_cols"]

    # Pydantic -> DataFrame
    data_admin_bo = pd.DataFrame([r.dict() for r in req.administration_bo.records])
    data_station =pd.DataFrame([r.dict() for r in req.station.records])

    # Pastikan semua fitur ada
    missing = [c for c in SELECTED_FEATURES_ADMIN_BO if c not in data_admin_bo.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Fitur berikut hilang di request: {missing}",
        )
    
    missing = [c for c in SELECTED_FEATURES_STATION if c not in data_station.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Fitur berikut hilang di request: {missing}",
        )

    # ADMINISTRATION BO
    df_cat = data_admin_bo[categorical_admin_bo]
    df_num = data_admin_bo[numeric_cols_admin_bo]

    df_cat_enc = encoder_admin_bo.transform(df_cat)
    enc_cols = encoder_admin_bo.get_feature_names_out(categorical_admin_bo)

    df_cat_enc_df = pd.DataFrame(df_cat_enc, columns=enc_cols, index=data_admin_bo.index)

    X_final = pd.concat([df_num, df_cat_enc_df], axis=1)

    preds_admin_bo = model_admin_bo.predict(X_final)
    preds_admin_bo = [float(p) for p in preds_admin_bo]

    # STATION

    df_cat = data_station[categorical_cols_station]
    df_num = data_station[numeric_cols_station]

    df_cat_enc = encoder_station.transform(df_cat)
    enc_cols = encoder_station.get_feature_names_out(categorical_cols_station)

    df_cat_enc_df = pd.DataFrame(df_cat_enc, columns=enc_cols, index=data_station.index)

    X_final = pd.concat([df_num, df_cat_enc_df], axis=1)

    preds_station = model_station.predict(X_final)
    preds_station = [float(p) for p in preds_station]

    return {
        "administration_bo": AdminBOPredictResponse(predictions=preds_admin_bo),
        "station": StationPredictResponse(predictions=preds_station)
    }

@app.post("/predict_payroll", response_model=PayrollPredictResponse)
def predict_payroll(req: PayrollPredictRequest):
    try:
        # Load Models
        art_cp = load_payroll_artifacts("cockpit")
        art_cb = load_payroll_artifacts("cabin")
        
        df = pd.DataFrame([r.dict() for r in req.records])
        
        # PREDICT COCKPIT
        enc_cp = art_cp["encoder"]
        df_cat_cp = pd.DataFrame(enc_cp.transform(df[CATEGORICAL_COLS_PAYROLL]), 
                                 columns=enc_cp.get_feature_names_out(CATEGORICAL_COLS_PAYROLL))
        df_num_cp = df[art_cp["numeric"]].reset_index(drop=True)
        X_cp = pd.concat([df_num_cp, df_cat_cp], axis=1)
        
        pred_log_cp = art_cp["model"].predict(X_cp)
        pred_real_cp = np.expm1(pred_log_cp) 
        
        # PREDICT CABIN 
        enc_cb = art_cb["encoder"]
        df_cat_cb = pd.DataFrame(enc_cb.transform(df[CATEGORICAL_COLS_PAYROLL]), 
                                 columns=enc_cb.get_feature_names_out(CATEGORICAL_COLS_PAYROLL))
        df_num_cb = df[art_cb["numeric"]].reset_index(drop=True)
        X_cb = pd.concat([df_num_cb, df_cat_cb], axis=1)
        
        pred_log_cb = art_cb["model"].predict(X_cb)
        pred_real_cb = np.expm1(pred_log_cb) 
        
        return {
            "cockpit_person_cost": [float(p) for p in pred_real_cp],
            "cabin_person_cost": [float(p) for p in pred_real_cb]
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_afgh", response_model=AFGHPredictResponse)
def predict_afgh(req: AFGHPredictRequest):
    try:
        artifacts_landing = load_landing_artifacts()
        artifacts_handling = load_handling_artifacts()
        artifacts_atc = load_atc_artifacts()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    model_landing = artifacts_landing["model"]
    encode_landing = artifacts_landing["encoder"]
    categorical_landing = artifacts_landing["categorical_cols"]
    numeric_cols_landing = artifacts_landing["numeric_cols"]

    model_handling = artifacts_handling["model"]
    encoder_handling = artifacts_handling["encoder"]
    categorical_cols_handling = artifacts_handling["categorical_cols"]
    numeric_cols_handling = artifacts_handling["numeric_cols"]

    model_atc = artifacts_atc["model"]
    encoder_atc = artifacts_atc["encoder"]
    categorical_cols_atc = artifacts_atc["categorical_cols"]
    numeric_cols_atc = artifacts_atc["numeric_cols"]


    # Pydantic -> DataFrame
    data_landing = pd.DataFrame([r.dict() for r in req.landing.records])
    data_handling =pd.DataFrame([r.dict() for r in req.handling.records])
    data_atc =pd.DataFrame([r.dict() for r in req.atc.records])

    # Pastikan semua fitur ada
    missing = [c for c in SELECTED_FEATURES_LANDING if c not in data_landing.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Fitur berikut hilang di request: {missing}",
        )
    
    missing = [c for c in SELECTED_FEATURES_HANDLING if c not in data_handling.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Fitur berikut hilang di request: {missing}",
        )
    
    missing = [c for c in SELECTED_FEATURES_ATC if c not in data_atc.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Fitur berikut hilang di request: {missing}",
        )

    # LANDING
    df_cat = data_landing[categorical_landing]
    df_num = data_landing[numeric_cols_landing]

    df_cat_enc = encode_landing.transform(df_cat)
    enc_cols = encode_landing.get_feature_names_out(categorical_landing)

    df_cat_enc_df = pd.DataFrame(df_cat_enc, columns=enc_cols, index=data_landing.index)

    X_final = pd.concat([df_num, df_cat_enc_df], axis=1)

    preds_landing = model_landing.predict(X_final)
    preds_landing = [float(p) for p in preds_landing]

    # HANDLING

    df_cat = data_handling[categorical_cols_handling]
    df_num = data_handling[numeric_cols_handling]

    df_cat_enc = encoder_handling.transform(df_cat)
    enc_cols = encoder_handling.get_feature_names_out(categorical_cols_handling)

    df_cat_enc_df = pd.DataFrame(df_cat_enc, columns=enc_cols, index=data_handling.index)

    X_final = pd.concat([df_num, df_cat_enc_df], axis=1)

    preds_handling = model_handling.predict(X_final)
    preds_handling = [float(p) for p in preds_handling]

    # ATC

    df_cat = data_atc[categorical_cols_atc]
    df_num = data_atc[numeric_cols_atc]

    df_cat_enc = encoder_atc.transform(df_cat)
    enc_cols = encoder_atc.get_feature_names_out(categorical_cols_atc)

    df_cat_enc_df = pd.DataFrame(df_cat_enc, columns=enc_cols, index=data_atc.index)

    X_final = pd.concat([df_num, df_cat_enc_df], axis=1)

    preds_atc = model_atc.predict(X_final)
    preds_atc = [float(p) for p in preds_atc]


    return {
        "landing": LandingPredictResponse(predictions=preds_landing),
        "handling": HandlingPredictResponse(predictions=preds_handling),
        "atc": ATCPredictResponse(predictions=preds_atc)
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
    
@app.post("/train_fc", response_model=FCTrainResponse)
def train_fc():
    """Latih ulang model dari file Excel."""
    try:
        metrics = train_fc_model()
        return FCTrainResponse(**metrics)
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

# =============================
# CREW FATA
# =============================

@app.post("/train_crew_fata", response_model=CrewFATATrainResponse)
def train_crew_fata():
    try:
        # Train Cockpit
        metrics_cockpit = train_single_crew_model("COCKPIT_CREW_TRAVEL", CF_COCKPIT_MODEL_PATH)
        global _cockpit_artifacts
        _cockpit_artifacts = joblib.load(CF_COCKPIT_MODEL_PATH)

        # Train Cabin
        metrics_cabin = train_single_crew_model("CABIN_CREW_TRAVEL", CF_CABIN_MODEL_PATH)
        global _cabin_artifacts
        _cabin_artifacts = joblib.load(CF_CABIN_MODEL_PATH)

        return {
            "cockpit": metrics_cockpit,
            "cabin": metrics_cabin
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# =============================
# ADMINISTRATION BO AND STATION 
# =============================
@app.post("/train_bofsc", response_model=BOFSCTrainResponse)
def train_bofsc():
    """Latih ulang model dari file Excel."""
    try:
        metrics_admin_bo = train_admin_bo_model()
        metrics_station = train_station_model()

        return {"administration_bo": AdminBOTrainResponse(**metrics_admin_bo),               
                "station": StationTrainResponse(**metrics_station)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#=============================================
# PAYROLL
#=============================================


@app.post("/train_payroll", response_model=PayrollTrainResponse)
def train_payroll():
    try:
        # Train Cockpit
        m_cockpit = train_single_payroll_model("COCKPIT_CREW_PERSON", FEATURES_PAYROLL_COCKPIT, PAYROLL_COCKPIT_MODEL_PATH)
        global _payroll_cockpit_artifacts
        _payroll_cockpit_artifacts = joblib.load(PAYROLL_COCKPIT_MODEL_PATH)
        
        # Train Cabin
        m_cabin = train_single_payroll_model("CABIN_CREW_PERSON", FEATURES_PAYROLL_CABIN, PAYROLL_CABIN_MODEL_PATH)
        global _payroll_cabin_artifacts
        _payroll_cabin_artifacts = joblib.load(PAYROLL_CABIN_MODEL_PATH)
        
        return {"cockpit": m_cockpit, "cabin": m_cabin}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# =============================
# AIRPORT FEES AND GROUND HANDLING
# =============================
@app.post("/train_afgh", response_model=AFGHTrainResponse)
def train_afgh():
    """Latih ulang model dari file Excel."""
    try:
        metrics_landing = train_landing_model()
        metrics_handling = train_handling_model()
        metrics_atc = train_atc_model()

        return {"landing": LandingTrainResponse(**metrics_landing),
                "handling": HandlingTrainResponse(**metrics_handling),
                "atc": ATCTrainResponse(**metrics_atc)}
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "traceback": tb
            }
        )


# ==================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run("ml_gi_api:app", host="0.0.0.0", port=8600, reload=True)