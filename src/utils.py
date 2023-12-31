import os
import sys 
import pandas as pd
import numpy as np 
from src.logger import logging
from src.exception import CustomException
import dill 
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score

def save_object(filepath, obj):
    try:
        logging.info('Entered the save_object method in Utils')
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)
        logging.info('Directory creation completed')
        
        with open(filepath, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        
        logging.info('Dumped the file')
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]
            # Training the model
            #model.fit(X_train, y_train)
            
            gs = GridSearchCV(model, para,cv = 3)
            gs.fit(X_train, y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
        
        logging.info('Evaluate model for loop successfully executed')
        logging.info(f'report = {report}')
        return report
    except Exception as e:
        raise CustomException(e, sys)