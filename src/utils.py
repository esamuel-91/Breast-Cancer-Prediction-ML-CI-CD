import os
import sys
import mlflow
import traceback
import pickle
from src.exception import CustomException
from src.logger import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def save_obj(file_path: str, obj):
    """
    This function is used to save the pickle file to the save location provided.
    arg1: File Path is string format
    arg2: Pickle File
    """
    with mlflow.start_run(nested=True):
        try:
            logger.info("Attempting to save the pickle file...")

            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)

            with open(file=file_path, mode="wb") as file_obj:
                pickle.dump(obj, file_obj)

            logger.info("Pickle File Successfully Saved... ")

        except Exception as e:
            mlflow.log_param("Pickle_Save_Exception", str(e))
            mlflow.log_text(
                "".join(traceback.format_exc()), "pickle_save_traceback.txt"
            )
            logger.error(f"Exception occured while trying to save the pickle file: {e}")
            raise CustomException(e, sys)


def load_obj(file_path: str):
    """
    This function is used to load the pickle file from the saved location.
    """
    with mlflow.start_run(nested=True):
        try:
            logger.info("Attempting to load the pickle file...")

            with open(file=file_path, mode="rb") as file_obj:
                load_pickle = pickle.load(file_obj)
                logger.info("Pickle File Successfully Loaded... ")

                return load_pickle
        except Exception as e:
            mlflow.log_param("Pickle_Load_Exception", str(e))
            mlflow.log_text(
                "".join(traceback.format_exc()), "pickle_load_traceback.txt"
            )
            logger.error(f"Exception occured while trying to load the object file: {e}")
            raise CustomException(e, sys)


def remove_outlier_iqr(data, column):
    """
    This function is used to remove the outlier from the feature column for the provided dataset.
    """
    with mlflow.start_run(nested=True):
        try:
            logger.info("Outlier Removal Started... ")

            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)

            IQR = Q3 - Q1

            lower_bound = Q1 - (1.5 * IQR)
            upper_bound = Q3 + (1.5 * IQR)

            clean_data = data[
                (data[column] >= lower_bound) & (data[column] <= upper_bound)
            ]

            logger.info(f"Outlier successfully removed from column: {column}")

            return clean_data
        except Exception as e:
            mlflow.log_param("Outlier_Removal_Exception", str(e))
            mlflow.log_text(
                "".join(traceback.format_exc()), "outlier_removal_traceback.txt"
            )
            logger.error(f"Exception occured while trying to remove the outlier: {e}")
            raise CustomException(e, sys)


def eval_model(X_train, X_test, y_train, y_test, models):
    """
    This function is used to evaluate the model on specific metrics
    """
    with mlflow.start_run(nested=True):
        try:
            report = {}

            for model_name, model in models.items():
                logger.info(f"{model_name} evaluation started....")
                model.fit(X_train, y_train)

                logger.info("Making prediction on test data..")
                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                class_report = classification_report(y_test, y_pred)
                confi_matrix = confusion_matrix(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred)

                report[model_name] = {
                    "Accuracy Score": accuracy,
                    "Classification Report \n": class_report,
                    "Confusion Matrix \n": confi_matrix,
                    "Roc Auc Score": roc_auc,
                }

                mlflow.log_metric(f"{model_name}_Accuracy Score", accuracy)
                mlflow.log_metric(f"{model_name}_roc_auc_score", roc_auc)

            logger.info(f"{model_name}_evaluation completed...")

            return report

        except Exception as e:
            mlflow.log_param("Model_Eval_Exception", str(e))
            mlflow.log_text("".join(traceback.format_exc()), "model_eval_traceback.txt")
            logger.error(f"Exception occured while trying to evaluate the model... {e}")
            raise CustomException(e, sys)
