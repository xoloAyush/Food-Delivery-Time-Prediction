import joblib
import pandas as pd
from xgboost import XGBRegressor

cols = [
    'Delivery_person_Age', 'Delivery_person_Ratings', 'distance_km',
    'order_hour', 'Road_traffic_density', 'Vehicle_condition',
    'multiple_deliveries', 'Type_of_order_Drinks', 'Type_of_order_Meal',
    'Type_of_order_Snack', 'Type_of_vehicle_electric_scooter',
    'Type_of_vehicle_motorcycle', 'Type_of_vehicle_scooter', 'Festival_Yes',
    'City_Semi-Urban', 'City_Urban', 'Weatherconditions_Fog',
    'Weatherconditions_Sandstorms', 'Weatherconditions_Stormy',
    'Weatherconditions_Sunny', 'Weatherconditions_Windy',
]

# Row 29851: Buffet, motorcycle, Metropolitian, Cloudy, no festival
row = {c: 0 for c in cols}
row.update({
    'Delivery_person_Age': 39, 'Delivery_person_Ratings': 4.6,
    'distance_km': 11.007735, 'order_hour': 17,
    'Road_traffic_density': 1, 'Vehicle_condition': 0,
    'multiple_deliveries': 1, 'Type_of_vehicle_motorcycle': 1,
})
df = pd.DataFrame([row])[cols]

scaler = joblib.load('standard_scaler.pkl')

model = XGBRegressor()
model.load_model('xgb_best.json')

scaled = scaler.transform(df)
print("scaled:", scaled[0][:7].round(4))
print("prediction:", model.predict(scaled)[0])