from src.models.predict import DeliveryTimePredictor

predictor = DeliveryTimePredictor()

sample_input = {
    "Delivery_person_Age": 30,
    "Delivery_person_Ratings": 4.7,
    "Vehicle_condition": "Good",
    "multiple_deliveries": 1,
    "distance_km": 5.2,
    "order_hour": 20,
    "order_minute": 30,
    "pickup_delay_minutes": 10,
    "order_day_of_week": 5,
    "is_weekend": 1,
    "is_peak_hour": 1,
    "Weather_conditions": "Fog",
    "Road_traffic_density": "Jam",
    "Type_of_order": "Snack",
    "Type_of_vehicle": "motorcycle",
    "Festival": "No",
    "City": "Metropolitian"
}

prediction = predictor.predict(sample_input)

print("Predicted Delivery Time:", prediction)