import os 
import sys
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_model

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr, preprocessor_path):
        try:

            # separate the target variable in the train and test arrays
            logging.info('Splitting training and testing input data')
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1],
            )

            # a list of models to experiment with
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbours Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            model_report:dict = evaluate_model(x_train=X_train, y_train=y_train, x_test=X_test, 
                                                y_test=y_test, models=models)


            ## to get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## to get best model froom dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = model_report[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No best Model Found!!!')
            logging.info('Found best model on both training and test data')

            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                            obj=best_model)

            predicted = best_model.predict(X_test)

            return r2_score(y_test, predicted)
            

        except Exception as e:
            raise CustomException(e, sys)