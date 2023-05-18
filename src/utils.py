import os
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import dill

def save_object(file_path,obj):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(models:dict,X_train,X_test,y_train,y_test):
    try:
        model_report = dict()

        for name,model in models.items():
            logging.info("processing model: {0}".format(name))
            reg = model.fit(X_train,y_train)
            y_pred = reg.predict(X_test)
            score = r2_score(y_test,y_pred)
            model_report.update({name:score})

        return model_report
    except Exception as e:
        raise CustomException(e,sys)


def evaluate_with_HP(models:dict,X_train,X_test,y_train,y_test,params):
    try:
        model_report = dict()

        for name,model in models.items():
            logging.info("Now Processing {0} model".format(name))
            grid = GridSearchCV(model,params[name],cv=5)
            grid.fit(X_train,y_train)
            model.set_params(**grid.best_params_)
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test,y_pred)
            model_report.update({name:score})

        return model_report

    except Exception as e:
        raise CustomException(e,sys)