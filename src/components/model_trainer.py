import os 
import sys 
from dataclasses import dataclass

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.utils import save_object, evaluate_models
from src.logger import logging
from src.exception import CustomException

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting Training and Test data')
            # We are splitting the input and output parameters in each array so that we can feed them to model
            X_train, y_train, X_test, y_test = (train_array[:,:-1], train_array[:,-1], test_array[:,:-1], test_array[:,-1])
            
            models = {
                "Random Forest Regressor": RandomForestRegressor(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K Neighbours Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            
            params = {
                "Decision Tree Regressor": 
                    {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_features' : ['sqrt', 'log2']
                    },
                "Random Forest Regressor":
                    {
                        'criterion':['squared_error','friedman_mse', 'absolute_error','poisson'],
                        'max_features':['sqrt', 'log2', None],
                        'n_estimators':[8, 16, 32, 64, 128, 256] 
                    },
                "Gradient Boosting Regressor":
                    {
                        'loss':['squared_error', 'huber', 'absolute_error', 'quantile']
                        'learning_rate':[0.1, 0.01, 0.05, 0.001],
                        'subsample':[0.6, 0.7, 0.75,0.8, 0.85, 0.9],
                        'criterion':['squared_error', 'friedman_mse'],
                        'n_estimators':[8, 16, 32, 64, 128, 256]
                    },
                "Linear Regression": {},
                "XGB Regressor":
                    {
                        'learning_rate':[0.1, 0.01, 0.05, 0.001],
                        'n_estimators':[8, 16, 32, 64, 128, 256]
                    }
                "AdaBoost Regressor":
                    {
                        "learning_rate":[0.01, 0.1, 0.05, 0.001],
                        'n_estimators':[8, 16, 32, 64, 128, 256]
                    }
            }
            # Get the best model score from dict
            model_report:dict = evaluate_models(X_train = X_train, y_train = y_train, X_test= X_test, y_test = y_test, models = models, params = params)
            
            # Get best model name from dict
            best_model_score = max(sorted(model_report.values()))
            logging.info(f'best_model_score = {best_model_score}')
            
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            logging.info(f'best_model_name = {best_model_name}')
            
            best_model = models[best_model_name]
            logging.info(f'best_model = {best_model}')
            
            if best_model_score < 0.06:
                raise CustomException('No best model found as all the scores are below 60%')
            
            logging.info(f'Best model found in both training and testing dataset')
            
            save_object(filepath = self.model_trainer_config.trained_model_file_path,
                        obj = best_model)
            
            predicted  = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            
            return r2_square
            
        except Exception as e:
            raise CustomException(e, sys)