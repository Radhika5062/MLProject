Machine Learning Project

ML Project
001 Video 
Create Github Repo
Open the folder in VS Code
Open terminal - create an environment - conda create -p vent python==3.8 -y
Activate the virtual environment -  conda activate venv/
Sync with GIT - git init, create a readme file. and then all the commands till end from GitHub. Before git push setup git configuration. 
Go to github - create gitignore file and then use python. Commit changes.
Now we will do git pull so that we can get gitignore file in our local system.
Create a new file - setup.py, requirements.txt. Why do we create setup,py file - so that we can build our application as a package and can use it anywhere. 
Setup.py code 
from setuptools import find_packages, setup
Setup metadata - name, version, author, author_email, packages, install_requires - this will tell all the requirememts. 
How setup.py will know about the packages that need to be used. We will use a src folder and then in that we will put __init__.py file and in all its sub folders too so that our setup.py file will be able to know the packages that are needed. All the directories or subdirectories which contain the __init__.py file are considered as packages and the find_packages function will be able to find it and use it. 
Entire project development happens in the src folder. 
We can’t add all the requirements in the form of lists in the install_requirements so we will create a function called get_requirements() and create definirtion for it and then we will call the requirements.txt file in it. Create this function and replace the requirements with this function call. Remember about .n being present in this file so we need to handle this scenario too. 
Now we always want our setup.py file to run so that it installs all the packages etc. We can either do this manually or add -e . in the requirements.txt file which will automatically trigger the setup.py file. 
Remember that -e .  should not come in the get_requirements call in the setup.py file as it is not needed. 
Now do pip install -r requirements.txt. This will create mlproject.egg-info file. 
Send this to git. 

002 Video
Currently in our src folder we only have __init__.py file.
Create a folder called components and then in it create the __init__.py file. 
In Compoment folder, create the data_ingestion.py file and then data_transformation.py file and model_trainer.py file. 
Create another folder called pipeline. Create a file called train_pipeline.py file in it and then predict_pipeline.py and __init__.py file so that this also becomes part of package. 
logger.py file at base level
exception.py at base level
utils.py. at base level.
exception.py file
import sys, 
def error_message_detail(error, error_detail:sys)
Exc_tb from error_detail.exc.info()
file_name = exc.tb.tb_frame.f_code.co_filename
Error message - error occurred in python script , line number, error message - file_name, exc.tb_lineno, error
return error message
class custom exception(Exception):
def __init__(self, error message, error_detail:sys):
Super.__init__(error_message)
self.error_message = call the function created above.
def __str__(self)
self.error_message
Logger.py
import logging, os
from datetime import date time
create a folder name - LOG_FILE = give a naming convention.log
create a logs folder path- logs_path = join the current working directory with “logs” and then the file name provided above.
Now it is time to create the above folder -> os.makedirs and give the above path name and then exist_ok as True
Now define the log file path - as a join between the “logs” folder and then log file. And this is the actual file where the errors wil be stored. 
When we need to write custom logs then we need to override the functionality from the logging module. This is done by setting some config in the basicConfig
filename = log file path
format = get the format “[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
level = info

003 Video
This is all about EDA and model training or selection.

004 Video
Now the first file we will select is data_ingestion.py.
import os
import sys
import our custom exception and logging that we have created.
import pandas
from sklearn.model_selection import train_test_split
from dataclasses import dataclass.
Create a class called DataIngestionConfig
use a decorator called data class. With this we can directly define the class variable and we will not need any init
define train_data_path: str = join the artifacts folder with train.csv file. Train data will be saved in this path.
repeat the above process with test data and then raw data.
These were the inputs we are giving to our data ingestion component so that it knows where to save these types of files
Create another class called DataIngestion.
As in this one we need to define variables as well as functions therefore we cannot use data class and will need to use init. 
Init function (self)
Call the above DataIngestionConfig class in this one so that all the three paths get created. self.ingestion_config = DataIngestionConfig()
Def initiate_data_ingestion
In this function we will try to bring data 
Logging - entered the data ingestion method or component.
In a try block. df - read the file from csv and give the path to the data file. Log read the data set as dataframe. Now as we know the paths of the folders that we will be saving the data too then now let us create those folders. Os,makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist = true). df.to_csv and give it raw data path and do index = flase and header = True. Now you fo train test split with this raw data. and then save the test set in test path and train set in train path. Now write logging as ingestion completed. Now return self.ingestion_config.train data path and test data path. 
In except - customexception as e, sys.
Now initiate using main and then call from terminal. 


005 Video
In this we will perform data transformation - data cleaning, feature engineering, convert categorical features into numerical features. 
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import columnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from logging and exception
import os
class DataTranformationConfig
When we will create any model then we will be saving it as a pickle file and we will require apath so create that path. preprocessor_obj_file_path = join artefacts with preprocessor.pkl file. 
As we are only declaring variables and are not creating any class functions hence we will use the dataclass decorator at the top. 
class DataTransformation
Def __init__(self)
self.data_configuration_config = DataTranformationConfig()
def get_data_transformer_object(self): This function will be handling converting categorical features into numerical features, performing standard scaler etc.This function is responsible for data transformation in the function definition. In try - 
define numerical features and then categorical features. Give names specifically. Numerical_columns, categorical_columns. 
define num_pipeline - pipeline, steps, 1. imputer, SimpleImputer, strategy - median. 2. Scaler - standardscaler. 
define cat_pipeline - Pipeline steps 1. imputer, simpleimputer, strategy - most_frequent. 2. One_hot_encoder, OneHotEncoder(), 3. scaler - standardscaler()
logging - display the columns names.
preprocessor = columnTransformer - num_pipeline, numerical columns, (name, pipeline, columns).similarly do this for categorical columns too. 
logging - Categorical columns encoding completed. numerical columns - scaling completed. 
return the preprocessor
except - raise customexception with e, sys. 
def initiate_data_transformation(self, train_path, test_path). train and test we are getting that from data ingestion. In try - 
Train_df = pd.read_Csv(train path)
Test_df = pd.read_csv(test_path)
logging - Reading train and test data completed.
logging - obtaining preprocessing object
Preprocessor_obj = self.get_data_transformer_object()
Target_column_name = ‘math_score’
copy paste - numerical_columns here
Input_feature_train_df = train_df.drop(columns = [target_coumn_name], axis = 1)
Target_feature_train_df = train_df[target_column_name]
do the above two steps for test too. 
log - Applying preprocessing ovject on training data frame and testing dataframe, 
Input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
do the same as above for test but only include transform as it is test and not train. 
Train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df]
do the same as above for test.
Log - Saved preprocessing obj
Save_object = filepath = self.data_transformation_config.preprocessor_obj_file_path, obj = preprocessing_obj. This function will be written in utils and is being called here. 
return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
UTILS = 
import os, sys, numpy, pandas, exception, dill
def save_object(filepath, obj). In try block add = 
Dir_path = os.path.dirname(filepath)
os.makedirs and send the above path and exist_ok = True
with open filepath in wb mode as file_obj = dill.dump(obj, file_obj)
raise exception in the except block. 
got back to transformation file - import utils- save_object
not go into data ingestion - 
import data transformation and both the classes. 
Data_transformation = DataTransformation()
Data_transformation.initiate_data_transformation(train and test data)
send to git. 



006 Video
In this one we will do the model trainer file. In this we will train different models and we will see accuracy and rscore. when you are training models then try all the models and then choose whichever is perfoming well. 
import os, sys
from dataclasses import dataclass
From catboost import CatBoostRegressor
from sklearn.ensemble import - AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import logging and custom exception
In requirements.txt - xgboost and cat boost
import save_object as well from utils.
Now as we have done with last two files - data ingestion and transformation, we first create config so we will do model trainer in similar manner. 
Use dataclass decorator
class - ModelTrainerConfig
create a variable - trained_model_file_path = join artifacts with model.pkl file. 
class ModelTrainer 
This class is responsible for training my model.
init class - > self.model_trainer_config = ModelTrainerConfig()
def initiate_model_trainer(self, train_array, test_array,) -> this is the output from the data transformation file.
In try block - log Splitting Training and Test input data
x_train, y_train, x_test, y_test = ( train_array[:, :-1], train_array[:,-1], test_array[:,:-1], test_array[:,-1]). In this we have separated the input and output features from the array so that we can feed then to model.
Now we will create a dictionary of models called models = Random Forest, Decision Treem Gradient Boosting, Linear Regression, K Neighbours Regressor, XGB Regressor, Catboost Regressor, AdaBoost Regressor
Currently we are not doing hyper parameter tuning.
Model_report : dict = evaluate_model(X_train = x_train, Y_train = y_train, y_test, models = models). this is a function that we will create in utils
Create function in utils.py - 
def - evaluate_models(X_train, y_train, X_test, y_test, models)
In try block. report  = {}
create a for loop for the length of list of models. In it, initialise model with its value. Then do model.fit on x train and y train which is our training the model. Now do y_train_pred = model.predict(x_train). Then do y_test_pred = model.preduct(x test).Then find out thr train_model_score using r2 score and giving values y train and y_train_pred. similarly do this for test_model_score. Now store these test model scores in report created as a dictionary of model name and these score values. Import r2 score.
Import evaluate models in the data modeler file.
Now get the best model score - first sort the values returned in model report and then apply max. 
now get the best model nameget keys of -> get the values of all models and then find the index of our selected best model. 
Now we need to get the name - models[best model name]
Check if best model score is less than 60% then raise a custom exception saying no best model found.
Logging - BEst model on both training and test dataset
 call save object and then save the model path.
Get the predicted model score. -> predicted = best_model( X_test)
To see the accuracy too - r2_square = r2_score(y_test, predicted)
return r2_square
In exception - raise custom exception. 
To test this - import model trainer file in data ingestion file. Import the other class too. 
In data ingestion - train_arr, test_arr,_= data transformation.iniate config. 
Initialise Model trainer.
Then call model trainer with initiate model trainer and give train array, test array from above step. Print this. This is the r2 score. 
Call data ingestion and then save to git. 

007 Video
Implement Hyperparameter tuning 