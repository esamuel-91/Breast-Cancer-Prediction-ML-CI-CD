import os
import sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
import mlflow
import traceback
from src.utils import save_obj
from src.exception import CustomException
from src.logger import logger
from sklearn.decomposition import PCA


@dataclass
class DataTransformationConfig:
    preprocessor_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self) -> None:
        self.transformation_config = DataTransformationConfig()

    def gather_transformation_obj(self):
        """
        This function is used to create a pipeline for Data Transformation.
        """
        try:
            logger.info("Creating Data Transformation Pipeline...")

            numerical_column = [
                "radius_mean",
                "texture_mean",
                "perimeter_mean",
                "area_mean",
                "smoothness_mean",
                "compactness_mean",
                "concavity_mean",
                "concave points_mean",
                "symmetry_mean",
                "fractal_dimension_mean",
                "radius_se",
                "texture_se",
                "perimeter_se",
                "area_se",
                "smoothness_se",
                "compactness_se",
                "concavity_se",
                "concave points_se",
                "symmetry_se",
                "fractal_dimension_se",
                "radius_worst",
                "texture_worst",
                "perimeter_worst",
                "area_worst",
                "smoothness_worst",
                "compactness_worst",
                "concavity_worst",
                "concave points_worst",
                "symmetry_worst",
                "fractal_dimension_worst",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("PCA", PCA(n_components=0.95)),
                ]
            )

            preprocessor = ColumnTransformer(
                [("num_pipeline", num_pipeline, numerical_column)]
            )

            logger.info("Pipeline Successfully Created...")

            return preprocessor
        except Exception as e:
            logger.error(
                f"Exception occured while creating transformation pipeline: {e}"
            )
            raise CustomException(e, sys)

    def initiate_transformation(self, train_set, test_set):
        """
        This function is used to intiate data transformation.
        """
        with mlflow.start_run(nested=True):
            try:
                train_df = pd.read_csv(train_set)
                test_df = pd.read_csv(test_set)

                logger.info("Using LabelEncoder to transform target feature...")
                le = LabelEncoder()
                train_df["diagnosis"] = le.fit_transform(train_df["diagnosis"])
                test_df["diagnosis"] = le.transform(test_df["diagnosis"])

                target = ["diagnosis"]
                drop_column = target

                input_feature_train_df = train_df.drop(drop_column, axis=1)
                input_target_train_df = train_df[target]

                input_feature_test_df = test_df.drop(drop_column, axis=1)
                input_target_test_df = test_df[target]

                logger.info("Obtaining Preprocessor Object...")
                preprocessor_obj = self.gather_transformation_obj()

                input_feature_train_arr = preprocessor_obj.fit_transform(
                    input_feature_train_df
                )
                input_feature_test_arr = preprocessor_obj.transform(
                    input_feature_test_df
                )

                train_arr = np.c_[
                    input_feature_train_arr, np.array(input_target_train_df)
                ]
                test_arr = np.c_[input_feature_test_arr, np.array(input_target_test_df)]

                logger.info("Data Transformation Completed Successfully...")

                save_obj(
                    file_path=self.transformation_config.preprocessor_file_path,
                    obj=preprocessor_obj,
                )

                return (train_arr, test_arr)

            except Exception as e:
                mlflow.log_param("Data_Transformation_Exception", str(e))
                mlflow.log_text(
                    "".join(traceback.format_exc()), "data_transformation_traceback.txt"
                )
                logger.error(
                    f"Exception occured while trying to transform the data: {e}"
                )
                raise CustomException(e, sys)
