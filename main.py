from pathlib import Path
from typing import Literal
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).parent

ml_model = {}

FEATURE_COLUMNS = [
    'Delivery_person_Age', 'Delivery_person_Ratings', 'distance_km',
    'order_hour', 'Road_traffic_density', 'Vehicle_condition',
    'multiple_deliveries', 'Type_of_order_Drinks', 'Type_of_order_Meal',
    'Type_of_order_Snack', 'Type_of_vehicle_electric_scooter',
    'Type_of_vehicle_motorcycle', 'Type_of_vehicle_scooter', 'Festival_Yes',
    'City_Semi-Urban', 'City_Urban', 'Weatherconditions_Fog',
    'Weatherconditions_Sandstorms', 'Weatherconditions_Stormy',
    'Weatherconditions_Sunny', 'Weatherconditions_Windy',
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = XGBRegressor()
    model.load_model(str(BASE_DIR / 'xgb_best.json'))
    ml_model['model'] = model
    ml_model['scaler'] = joblib.load(BASE_DIR / 'standard_scaler.pkl')
    yield
    ml_model.clear()


app = FastAPI(title="Food Delivery Time Prediction", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class DeliveryOrder(BaseModel):
    Delivery_person_Age: float = Field(ge=15, le=80)
    Delivery_person_Ratings: float = Field(ge=1, le=5)
    distance_km: float = Field(gt=0, le=100)
    order_hour: int = Field(ge=0, le=23)
    Road_traffic_density: int = Field(ge=0, le=3)
    Vehicle_condition: int = Field(ge=0, le=3)
    multiple_deliveries: float = Field(ge=0, le=5)
    Type_of_order: Literal["Snack", "Meal", "Drinks", "Buffet"]
    Type_of_vehicle: Literal["motorcycle", "scooter", "electric_scooter", "bicycle"]
    Festival: Literal["Yes", "No"]
    City: Literal["Urban", "Semi-Urban", "Metropolitian"]
    Weatherconditions: Literal["Sunny", "Cloudy", "Windy", "Fog", "Stormy", "Sandstorms"]


def build_features(d: DeliveryOrder) -> pd.DataFrame:
    row = {
        'Delivery_person_Age': d.Delivery_person_Age,
        'Delivery_person_Ratings': d.Delivery_person_Ratings,
        'distance_km': d.distance_km,
        'order_hour': d.order_hour,
        'Road_traffic_density': d.Road_traffic_density,
        'Vehicle_condition': d.Vehicle_condition,
        'multiple_deliveries': d.multiple_deliveries,
        f'Type_of_order_{d.Type_of_order}': 1,
        f'Type_of_vehicle_{d.Type_of_vehicle}': 1,
        f'Festival_{d.Festival}': 1,
        f'City_{d.City}': 1,
        f'Weatherconditions_{d.Weatherconditions}': 1,
    }
    # Baseline categories (Buffet, bicycle, Festival_No, Metropolitian, Cloudy)
    # have no column, so they end up as all zeros, same as in training.
    return pd.DataFrame([row]).reindex(columns=FEATURE_COLUMNS, fill_value=0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post('/predict')
def predict(data: DeliveryOrder):
    input_df = build_features(data)

    # Same two steps as the notebook: scale, then predict
    scaled = ml_model['scaler'].transform(input_df)
    scaled_df = pd.DataFrame(scaled, columns=FEATURE_COLUMNS)

    prediction = float(ml_model['model'].predict(scaled_df)[0])
    return {"predicted_delivery_time_min": round(prediction, 2)}


# Keep this last so it doesn't swallow the API routes
app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")