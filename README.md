# Zomato-Delivery-Time-Prediction

# 🍽️ Zomato Delivery Time Prediction

Predict the delivery time of food orders using Machine Learning based on factors such as distance, traffic, weather, and delivery partner attributes.

This project demonstrates a complete **end-to-end MLOps pipeline**, including data preprocessing, model training, experiment tracking with MLflow, and deployment through a Dockerized FastAPI application.

---

## 📌 Project Overview

The system predicts **estimated delivery time** for Zomato orders using historical delivery data.

The prediction considers several operational factors such as:

- Delivery partner age and ratings
- Distance between restaurant and customer
- Road traffic density
- Weather conditions
- Type of vehicle used
- Type of order
- Festival indicators
- City information

The trained model is served through a **FastAPI web interface** for real-time predictions.

---

## 🚀 Key Features

- End-to-end machine learning pipeline
- Feature engineering for delivery time prediction
- Multiple model training and evaluation
- Experiment tracking using **MLflow**
- Model serialization for inference
- **FastAPI prediction service**
- **Dockerized deployment**
- Ready for **AWS cloud deployment**

---

## 🤖 Machine Learning Models

The following regression models were evaluated:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

### Evaluation Metrics

Models were evaluated using:

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

The best model was selected automatically using **cross-validation RMSE**.

---

## 📊 Experiment Tracking (MLflow)

All model experiments are tracked using **MLflow**, which logs:

- Model parameters
- Evaluation metrics
- Model artifacts

### Run MLflow UI

```bash
mlflow ui
