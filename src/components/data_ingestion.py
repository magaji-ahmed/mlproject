import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


# use a dataclass decorator to create a class specifically for storing data 
# there is no need to define a constructor using init function
@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        # the config class stores the data we need
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method')
        try:
            # read the data from the csv, but can be from a database
            df = pd.read_csv('Notebook\data\stud.csv')
            logging.info('read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # put the raw data file into the raw path in the artifacts folder
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info('Train Test Split initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # put the train and test data files into their respective paths in the artifacts folder
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of data is complete')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
           raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()