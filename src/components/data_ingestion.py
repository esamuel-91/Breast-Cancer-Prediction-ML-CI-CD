import os
import sys
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logger
import mlflow
import traceback
import pandas as pd
from dataclasses import dataclass
from src.utils import remove_outlier_iqr


@dataclass
class DataIngestionConfig:
    train_file_path: str = os.path.join("artifacts", "train.csv")
    test_file_path: str = os.path.join("artifacts", "test.csv")
    raw_file_path: str = os.path.join("artifacts", "raw.csv")


class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_ingestion(self):
        """
        This function is used to ingest the data and split the data using train test split.
        It will save the split data into the artifacts folder created in the local dir
        """
        with mlflow.start_run(nested=True):
            try:
                logger.info("Data Ingestion Started...")
                logger.info("Reading the Dataframe...")

                df = pd.read_csv(
                    "https://raw.githubusercontent.com/gscdit/Breast-Cancer-Detection/refs/heads/master/data.csv"
                )
                logger.info(f"Shape of the dataframe: {df.shape}")

                logger.info("Removing unwanted columns...")
                df.drop(["id", "Unnamed: 32"], axis=1, inplace=True)

                logger.info("Saving the Raw data into the artifacts folder...")
                os.makedirs(
                    os.path.dirname(self.ingestion_config.raw_file_path), exist_ok=True
                )

                df.to_csv(self.ingestion_config.raw_file_path, index=False)

                logger.info("Removing outlier from the dataframe...")

                numeric_cols = df.select_dtypes(include=["number"]).columns.to_list()
                for col in numeric_cols:
                    df = remove_outlier_iqr(data=df, column=col)

                logger.info(
                    f"Shape of the dataframe after remove the outlier: {df.shape}"
                )

                logger.info(
                    "Using Train Test Split to split the dataframe and save to the artifacts folder..."
                )
                train_set, test_set = train_test_split(
                    df, test_size=0.20, stratify=df["diagnosis"],random_state=42
                )

                train_set.to_csv(
                    self.ingestion_config.train_file_path, index=False, header=True
                )
                test_set.to_csv(
                    self.ingestion_config.test_file_path, index=False, header=True
                )

                mlflow.log_artifact(
                    self.ingestion_config.raw_file_path, "ingestion/raw"
                )
                mlflow.log_artifact(
                    self.ingestion_config.train_file_path, "ingestion/train"
                )
                mlflow.log_artifact(
                    self.ingestion_config.test_file_path, "ingestion/test"
                )
                logger.info("Data Ingestion Completed Successfully...")

                return (
                    self.ingestion_config.train_file_path,
                    self.ingestion_config.test_file_path,
                )

            except Exception as e:
                mlflow.log_param("Data_Ingestion_Exception", str(e))
                mlflow.log_text(
                    "".join(traceback.format_exc()), "data_ingestion_traceback.txt"
                )
                logger.error(f"Exception occured while trying to ingest the data: {e}")
                raise CustomException(e, sys)
