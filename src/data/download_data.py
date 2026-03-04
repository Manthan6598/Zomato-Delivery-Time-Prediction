import os
import kaggle
from src.utils.logger import logger


DATASET_NAME = "saurabhbadole/zomato-delivery-operations-analytics-dataset"

DOWNLOAD_PATH = "dataset/raw"

def download_dataset():
    try:
        logger.info("Starting dataset download from Kaggle....")

        os.makedirs(DOWNLOAD_PATH,exist_ok=True)
        kaggle.api.dataset_download_files(
            DATASET_NAME,
            path = DOWNLOAD_PATH,
            unzip = True
        )
        logger.info("Dataset downloaded successfully..!")

    except Exception as e:

        logger.info(f"Error in downloading the dataset.. {e}")

        raise(e)
    
if __name__ == "__main__":

    download_dataset()

