import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler


from src.components.data_ingestion import DataIngestion
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation Initiated")

            # Define which columns should be ordinal_encoded and which should be scaled
            # In this dataset their is no categorical columns is present so no required

            num_columns = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2',
       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
            
            logging.info("Pipeline Initiated")

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy = 'median')),
                    ('scaler', StandardScaler())

                ]
            )

            # Preprocessor
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, num_columns)
                ]
            )

            return preprocessor

            logging.info("Pipeline Completed")

        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Reading Train & Test Data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read Train & Test Data Completed")
            logging.info(f"Train Dataframe Head :\n{train_df.head().to_string()}")
            logging.info(f"Test Dataframe Head : \n{test_df.head().to_string()}")

            logging.info("Obtaining Preprocessing Object")

            preprocessor_obj = self.get_data_transformation_object()

            target_column_name = "default.payment.next.month"

            drop_column_name = [target_column_name, "ID"]

            input_feature_train_df = train_df.drop(columns = drop_column_name, axis = 1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns = drop_column_name, axis = 1)
            target_feature_test_df = test_df[target_column_name]

            # Transforming Using Preprocessor Object
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)

            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            logging.info("Applying Preprocessing Object On Training & Testing Datasets.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            logging.info("Preprocessor Pickle File Saved")
            logging.info(f"Train_arr_shape : {train_arr.shape}")
            logging.info(f"Test_arr_shape : {test_arr.shape}")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path    
            )
        except Exception as e:
            logging.info("Exception Occured in the Initiate_data_transformation")

            raise CustomException(e, sys)
        


if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
            



