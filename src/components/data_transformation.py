import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

@dataclass
class data_transformation_config:
    data_transform_config = os.path.join('artifacts',"preprocessor.pkl")

class Data_Transformation:
    def __init__(self):
        self.data_config = data_transformation_config()

    def get_data_transformation(self):
        try:
            numerical = ['writing_score','reading_score']
            categorical = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']

            num_pipeline = Pipeline(
                steps=[
                    ('impute',SimpleImputer(strategy='median')),
                    ('scaling',StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('impute',SimpleImputer(strategy='most_frequent')),
                    ('encoding',OneHotEncoder()),
                    ('scailing',StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ('Numerical Features',num_pipeline,numerical),
                    ('Categorical Features',cat_pipeline,categorical)
                ]
            )

            return preprocessor
    
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info("data transformation started")
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)

            preprocessor_obj = self.get_data_transformation()

            logging.info("preprocessor object created")
            target = 'math_score'
            independent_train_features = df_train.drop(columns=target,axis=1)
            independent_test_features = df_test.drop(columns=target,axis=1)

            dependent_train_feature = df_train[target]
            dependent_test_feature = df_test[target]

            independent_train_features_arr = preprocessor_obj.fit_transform(independent_train_features)
            independent_test_features_arr = preprocessor_obj.transform(independent_test_features)

            train_set = np.c_[independent_train_features_arr,np.array(dependent_train_feature)]
            test_set = np.c_[independent_test_features_arr,np.array(dependent_test_feature)]

            logging.info("fit transform and transform done and concatenation and creating datafrom done now saving objects")

            save_object(
                file_path=self.data_config.data_transform_config,
                obj = preprocessor_obj
            )

            logging.info("object saved now returning the train and test dataset")

            return (
                train_set,
                test_set,
                preprocessor_obj
            )

        except Exception as e:
            raise CustomException(e,sys)