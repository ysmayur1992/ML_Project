import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
from src.components.data_transformation import data_transformation_config
from src.components.data_transformation import Data_Transformation
from src.components.model_trainer import model_trainer_config
from src.components.model_trainer import Model_Training


@dataclass
class data_ingestion_config:
    raw_data_path:str = os.path.join('artifacts',"data.csv")
    train_data_path:str = os.path.join('artifacts',"train.csv")
    test_data_path:str = os.path.join('artifacts',"test.csv")

class Data_Ingestion:
    def __init__(self):
        self.data_ingestion_config = data_ingestion_config()

    def initiate_data_ingestion(self):
        try:
            logging.info("data ingestion started")
            df = pd.read_csv('notebook\data\stud.csv')
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.data_ingestion_config.raw_data_path,index=0,header=1)

            train,test = train_test_split(df,test_size=0.2,random_state=42)

            train.to_csv(self.data_ingestion_config.train_data_path,index=0,header=1)
            test.to_csv(self.data_ingestion_config.test_data_path,index=0,header=1)
            logging.info("data ingestion completed now returning train and test file paths")
            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )


        except Exception as e:
            raise CustomException(e,sys)



if __name__=="__main__":
    data_ingestion = Data_Ingestion()
    train_path,test_path = data_ingestion.initiate_data_ingestion()

    data_transform = Data_Transformation()
    train_set,test_set,_ = data_transform.initiate_data_transformation(train_path,test_path)

    regression_model_obj = Model_Training()
    regression_model,score = regression_model_obj.initiate_model_training(train=train_set,test=test_set)

    print("The actual score of the model is {0}".format(score*100))