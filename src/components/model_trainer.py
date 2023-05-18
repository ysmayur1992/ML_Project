import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models,evaluate_with_HP
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

@dataclass
class model_trainer_config:
    model_training_config = os.path.join('artifacts',"model_trainer.pkl")

class Model_Training:
    def __init__(self):
        self.config = model_trainer_config()

    def initiate_model_training(self,train,test):
        try:
            logging.info("Model training initiated...")

            X_train,X_test,y_train,y_test = (
                train[:,:-1],
                test[:,:-1],
                train[:,-1],
                test[:,-1]
            )

            model_dict = {
                "Linear Regression":LinearRegression(),
                "Decision Tree Regression":DecisionTreeRegressor(),
                "XGBoost Regression":XGBRegressor(),
                "CatBoost Regression":CatBoostRegressor(verbose=False),
                "AdaBoost Regression":AdaBoostRegressor(),
                "Gradient Boost Regression":GradientBoostingRegressor(),
                "Random Forest Regression":RandomForestRegressor()
            }

            logging.info("dictionary created now evaluating the models and developing scores")

            '''model_with_scores = evaluate_models(model_dict,X_train,X_test,y_train,y_test)
            sorted_model = dict(sorted(model_with_scores.items(),key = lambda x:x[1]))

            model_name = list(sorted_model.keys())[0]
            model_score = list(sorted_model.values())[0]

            message = "the model name is {0} and its accuracy score is {1} percent".format(model_name,model_score*100)
            print(message)

            save_object(
                file_path=self.config.model_training_config,
                obj = model_dict[model_name]
            )

            logging.info("object saved now returning score")
            return (
                model_dict[model_name],
                model_score*100
            )
            '''

            parameter = {
                "Linear Regression":{},
                "Decision Tree Regression":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "XGBoost Regression":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoost Regression":{
                    'depth':[6,8,10],
                    'learning_rate':[0.1,0.01,0.05,0.001],
                    'iterations':[30,50,100]
                },
                "AdaBoost Regression":{
                    'learning_rate':[0.1,0.01,0.05,0.001],
                    'n_estimators':[8,16,32,64,128,256]
                },
                "Gradient Boost Regression":{
                    'learning_rate':[0.1,0.01,0.05,0.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators':[8,16,32,64,128,256]
                },
                "Random Forest Regression":{
                    'criterion':['squared_error','friedman_mse','absolute_error','poisson']
                }
            }

            optimised_models_score = evaluate_with_HP(model_dict,X_train,X_test,y_train,y_test,parameter)

            sorted_op_models = dict(sorted(optimised_models_score.items(),key= lambda item:item[1],reverse=True))

            model_name = list(sorted_op_models.keys())[0]
            model_score = list(sorted_op_models.values())[0]

            message = "the model name is {0} and its accuracy score is {1} percent".format(model_name,model_score*100)
            print(message)

            logging.info("now saving object and returning score and model")

            save_object(
                obj = model_dict[model_name],
                file_path=self.config.model_training_config
            )

            return (
                model_dict[model_name],
                model_score
            )
        
        except Exception as exc:
            raise CustomException(exc,sys)