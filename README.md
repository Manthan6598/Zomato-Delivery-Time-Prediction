# Zomato Delivery Time Prediction

An end-to-end **Machine Learning + MLOps project** that predicts food delivery time based on operational, environmental, and delivery partner features.

This project demonstrates how a machine learning model can be trained, tracked, and deployed as a production-ready API using **FastAPI, MLflow, and Docker**.

---

## Project Overview

Food delivery platforms must estimate delivery time accurately to improve customer satisfaction and operational efficiency.

Delivery time is influenced by multiple factors such as:

- Distance between restaurant and customer
- Traffic density
- Weather conditions
- Delivery partner ratings
- Vehicle type
- Number of simultaneous deliveries

This project builds a **machine learning system that predicts delivery time in minutes** using historical delivery data and deploys the model through an API.

---

## Key Features

- End-to-end machine learning pipeline
- Feature engineering for delivery prediction
- Multiple regression models trained and evaluated
- Experiment tracking using **MLflow**
- Model deployment using **FastAPI**
- Containerized using **Docker**
- Cloud deployment ready (AWS)

---

## Dataset

The dataset contains operational information related to food deliveries.

### Input Features

| Feature | Description |
|------|-------------|
| Delivery_person_Age | Age of delivery partner |
| Delivery_person_Ratings | Rating of delivery partner |
| Vehicle_condition | Condition of vehicle |
| multiple_deliveries | Number of deliveries handled simultaneously |
| distance_km | Distance between restaurant and customer |
| order_hour | Hour of order placement |
| order_minute | Minute of order placement |
| pickup_delay_minutes | Delay during order pickup |
| order_day_of_week | Day when order was placed |
| Weather_conditions | Weather during delivery |
| Road_traffic_density | Traffic conditions |
| Type_of_order | Type of order |
| Type_of_vehicle | Delivery vehicle used |
| Festival | Festival indicator |
| City | City where delivery occurs |

### Target Variable

Delivery time (minutes)

---

## Feature Engineering

Additional features were created to capture real-world delivery patterns.

Examples:

- **is_weekend** → Indicates if the order was placed on weekend
- **is_peak_hour** → Indicates high demand delivery hours

Peak hours considered:

- 12 PM – 1 PM
- 7 PM – 9 PM

---

## Machine Learning Models

Several regression models were trained and evaluated:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

Models were compared using **cross-validation and test performance metrics**.

---

## Model Evaluation

Evaluation metrics used:

- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score

### Best Model

**XGBoost Regressor**

Performance:

- RMSE: ~3.8
- MAE: ~3.0
- R² Score: ~0.83

---

## Experiment Tracking

All experiments were tracked using **MLflow**, which logs:

- Model parameters
- Training metrics
- Evaluation results
- Model artifacts

Run MLflow locally:
```
mlflow ui
```

---

## Model Deployment

The trained model is deployed using **FastAPI** to provide real-time predictions.

### API Endpoints

| Endpoint   | Method | Description                     |
|------------|--------|---------------------------------|
| `/`        | GET    | Loads prediction interface      |
| `/predict` | POST   | Returns predicted delivery time |

---

## Running the Project Locally

### Clone the Repository
```
git clone https://github.com/Manthan6598/Zomato-Delivery-Time-Prediction.git
cd Zomato-Delivery-Time-Prediction
```

### Install Dependencies

```
pip install -r requirements.txt
```


### Run FastAPI Server

```
uvicorn app:app --reload
```


---

## Docker Deployment

Build Docker image:

```
docker build -t zomato-delivery-api .
```

Run the container:

```
docker run -p 8000:8000 zomato-delivery-api
```

Open the Browser

```
http://localhost:8000
```

---

## Project Structure : 

<img width="456" height="390" alt="image" src="https://github.com/user-attachments/assets/782592d5-e008-4572-bb20-a9786f0d6450" />

## Project Architecture :

<img width="484" height="244" alt="image" src="https://github.com/user-attachments/assets/bed0c082-78a6-4d38-84df-64cc70e7713b" />

AWS EC2 Deployed Link : 

Project Preview:

Preview 1:
<img width="1919" height="1026" alt="image" src="https://github.com/user-attachments/assets/e008da44-0581-41e6-bf6b-1a6365d7f5fe" />

Preview 2:
<img width="1913" height="1027" alt="image" src="https://github.com/user-attachments/assets/008b5319-f380-4363-be34-d6b81f22f142" />


