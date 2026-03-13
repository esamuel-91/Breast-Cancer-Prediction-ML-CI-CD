import os
import sys
from src.exception import CustomException
from src.logger import logger
from src.utils import save_obj
from dataclasses import dataclass
from src.utils import eval_model
import mlflow
import traceback
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


@dataclass
class ModelTrainerConfig:
    model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self) -> None:
        self.trainer_config = ModelTrainerConfig()

    def initiate_trainer(self, train_arr, test_arr):
        """
        This function is used to initiate the model trainer and train all the models.
        """
        with mlflow.start_run(nested=True):
            try:
                logger.info("Model Trainer Initiated...")

                X_train, X_test, y_train, y_test = (
                    train_arr[:, :-1],
                    test_arr[:, :-1],
                    train_arr[:, -1],
                    test_arr[:, -1],
                )

                models = {
                    "LogisticRegression": LogisticRegression(
                        C=1.0, penalty="l1", solver="saga"
                    ),
                    "SVC": SVC(C=1.2, kernel="rbf",probability=True),
                    "DecisionTreeClassifier": DecisionTreeClassifier(
                        criterion="entropy",
                        max_depth=30,
                        min_samples_leaf=3,
                        min_samples_split=5,
                        splitter="random",
                    ),
                    "RandomForestClassifier": RandomForestClassifier(
                        criterion="entropy",
                        max_depth=20,
                        min_samples_leaf=1,
                        min_samples_split=5,
                        n_estimators=100,
                    ),
                }

                report = eval_model(X_train, X_test, y_train, y_test, models)
                best_model_name = max(
                    report, key=lambda model_name: report[model_name]["Roc Auc Score"]
                )
                best_model = models[best_model_name]
                best_model_score = report[best_model_name]["Roc Auc Score"]

                print(
                    f"Best Model found: {best_model} and best model score is: {best_model_score}"
                )
                print(
                    "\n--------------------------------------------------------------------------\n"
                )

                save_obj(file_path=self.trainer_config.model_file_path, obj=best_model)

                mlflow.log_artifact(
                    self.trainer_config.model_file_path, artifact_path="model.pkl"
                )
                mlflow.log_param("Best Model", best_model)
                mlflow.log_param("Best Model Score", best_model_score)

                logger.info("Model Training Completed Successfully...")

            except Exception as e:
                mlflow.log_param("Model_Trainer_Exception", str(e))
                mlflow.log_text(
                    "".join(traceback.format_exc()), "model_trainer_traceback.txt"
                )
                logger.error(f"Exception occured while training the model: {e}")
                raise CustomException(e, sys)
