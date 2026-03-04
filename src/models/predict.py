import joblib
import pandas as pd
from src.utils.logger import logger

MODEL_PATH = "models/delivery_time_model.pkl"

class DeliveryTimePredictor:

    def __init__(self):

        logger.info("Loading Model...")

        self.model = joblib.load(MODEL_PATH)

        self.feature_names = self.model.named_steps["preprocessor"].feature_names_in_

        logger.info("Model Loaded Successfully")

    def predict(self, input_data):

        try:

            logger.info("Creating DataFrame from input")

            df = pd.DataFrame([input_data])

            df = df[self.feature_names]

            logger.info("Prediction in progress...")

            prediction = self.model.predict(df)

            logger.info(f"Prediction Successful: {prediction[0]}")

            return float(prediction[0])

        except Exception as e:

            logger.error(f"Prediction Failed: {str(e)}")

            raise e