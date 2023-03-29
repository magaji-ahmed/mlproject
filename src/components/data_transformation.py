# the main purpose of the data transformation file is to do feature engineering
# Data cleaning, change something about the data set, convert categorical features
# into numerical features along with other forms of data transforms needed by your 
# models such as hangling missing values

import sys 
import os
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str = os.path.join('artifact','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_obj(self):

        """
        This function is responsible for data transformation
        """

        try:
            num_features = ['writing_score','reading_score']
            cat_features = ['gender','race_ethnicity','parental_level_of_education',
                                'lunch','test_preparation_course']

            # create a pipeline and handle missing values within the pipeline
            # fill in the missing values and scale the values
            num_pipeline = Pipeline(
                steps=[
                    ('Imputer', SimpleImputer(strategy='median')), # use median due to outliers in data
                    ('Scaler', StandardScaler())
                ]
            )

            # fill in the missing values, do one hot encoding because column space is relatively low
            # then scale the values
            cat_pipline = Pipeline(
                steps=[
                    ('Imputer', SimpleImputer(strategy='most_frequent')), # using mode because its categorical
                    ('One_Hot_Encoder', OneHotEncoder()),
                    ('Scaler', StandardScaler())
                ]
            )

            logging.info(f'Numerical columns standard scaling completed for {num_features}')
            logging.info(f'Categorical columns encoding completed for {cat_features}')

            preprocessor = ColumnTransformer(
                [
                    ('numerical pipepline', num_pipeline, num_features),
                    ('Categorical pipeline', cat_pipline, cat_features)
                ]
            )

            logging.info('proprocessor build complete')

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transform(self, train_path, test_path):
        try:
            # read data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('read train and test data complete')

            logging.info('Obtaing preprocessing object')
            preprocessor_obj = self.get_data_transformer_obj()

            target_column_name = 'math_score'

            num_features = ['writing_score','reading_score']

            input_feature_train_df = train_df(columns=target_column_name, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df(columns=target_column_name, axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info('Applying preprocessor object to train and test dataframes')
            
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info('Saved preprocessing object')
            # save the preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessing_object_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            

        except Exception as e:
            pass