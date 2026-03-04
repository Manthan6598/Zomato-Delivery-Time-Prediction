import os
import pandas as pd
from src.utils.logger import logger

RAW_DATA_PATH = "dataset/raw"
PROCESSED_DATA_PATH = "data/processed"

def load_raw_data():
    try:
        logger.info("Start Data Ingestion....")
        files = [f for f in os.listdir(RAW_DATA_PATH) if(f.endswith(".csv") or (f.endswith(".xlsx")))]
        if not files:
            raise Exception("No Files Present in Your Folder....!")
        
        file_path = os.path.join(RAW_DATA_PATH,files[0])
        logger.info(f"Loading file: {file_path}")
        df = pd.read_csv(file_path)

        logger.info("Data Loaded Successfully...!")
        logger.info("Shape: {df.shape}")

        return df

    except Exception as e:
        logger.error(f"Error in data ingestion: {e}")

        raise e
    
def save_processed_data(df):
    try:
        os.makedirs(PROCESSED_DATA_PATH,exist_ok=True)
        output_path = os.path.join(PROCESSED_DATA_PATH,"processed_data.csv")
        df.to_csv(output_path,index=False)
        logger.info(f"Processed data saved at {output_path}")

    except Exception as e:
        logger.error(f"Error saving processed data : {e}")

        raise e
    
if __name__ == "__main__":
    df = load_raw_data()
    save_processed_data(df)
