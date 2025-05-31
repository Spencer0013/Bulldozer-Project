import os
import sys
from src.exception import CustomException  # Custom error handler
from src.logger import logging              # Custom logger setup
import pandas as pd

from dataclasses import dataclass




@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")  # Path for saving training data
    val_data_path: str = os.path.join('artifacts', "val.csv")  # Path for saving training data
    test_data_path: str = os.path.join('artifacts', "test.csv")    # Path for saving test data
    raw_data_path: str = os.path.join('artifacts', "data.csv")     # Path for saving raw/full data

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            
            df = pd.read_csv(r'C:\Users\ainao\Downloads\Projects\Bulldozer Project\data\bulldozer\TrainAndValid.csv',low_memory=False,parse_dates=['saledate'])
            df_test = pd.read_csv(r'C:\Users\ainao\Downloads\Projects\Bulldozer Project\data\bulldozer\Test.csv',low_memory=False,parse_dates=['saledate'])
            logging.info('Read the dataset as dataframe')

            # Create directory for artifacts if not exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Sort dataset in order of saledate
            df.sort_values(by=['saledate'], inplace=True, ascending=True)
    

            #Add datetime parameters for saledate column
            df["saleYear"] = df.saledate.dt.year
            df["saleMonth"] = df.saledate.dt.month
            df["saleDay"] = df.saledate.dt.day
            df["saleDayOfWeek"] = df.saledate.dt.dayofweek
            df["saleDayOfYear"] = df.saledate.dt.dayofyear

            #Remove saledate column
            df.drop('saledate', axis=1, inplace=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
           

        
            train_set = df[df.saleYear != 2012]
            val_set = df[df.saleYear==2012]

            # Save train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            val_set.to_csv(self.ingestion_config.val_data_path, index=False, header=True)
            df_test.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            
            logging.info(f"Train data saved to: {self.ingestion_config.train_data_path}")
            logging.info(f"Validation data saved to: {self.ingestion_config.val_data_path}")
            logging.info(f"Test data saved to: {self.ingestion_config.test_data_path}")

            logging.info(f"Train set size: {train_set.shape}")
            logging.info(f"Validation set size: {val_set.shape}")
            logging.info(f"Test set size: {df_test.shape}")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.val_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, val_data, test_data = obj.initiate_data_ingestion()
