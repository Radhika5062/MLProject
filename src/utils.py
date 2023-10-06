import os
import sys 
import pandas as pd
import numpy as np 
from src.logger import logging
from src.exception import CustomException
import dill 

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