import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model
from dataclasses import dataclass
import sys
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artefacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test')
            X_train,y_train,X_test,y_test=(train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])

            #Train Multiple models
            models={
                    'LinearRegression':LinearRegression(),
                    'Lasso':Lasso(),
                    'Ridge':Ridge(),
                    'ElasticNet':ElasticNet()
                   }
            
            #Calling the evaluate_model function 
            model_report=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('='*135)
            logging.info(f'Model Report :{model_report}')

            #Selecting the best model based on the r2_score 
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            # print('The Model Report :', model_report)
            # print('This is where error is occuring best_model_score :::: ', best_model_score)
            # print('This is where error is occuring best_model_name :::: ', best_model_name)

            best_model=models[best_model_name]
            print(f'Best Model is : {best_model_name}, its R2_score is : {best_model_score}')
            print('='*135)
            logging.info(f'Best Model is : {best_model_name}, its R2_score is : {best_model_score}')

            #Calling the save_object function to save the model.pkl file
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

        except Exception as e:
            logging.info('Exception occured in Model Training')
            raise CustomException(e,sys)
        