import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    """
    Abstract Class defining strategy for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreprocessStrategy(DataStrategy):
    """
    Data preprocessing strategy which preprocesses the data.
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            """
                Performs transformations on df and returns transformaed df.
                """
            # Convert 'saledate' column to datetime
            data['saledate'] = pd.to_datetime(data['saledate'])
            data["saleYear"] = data.saledate.dt.year
            data["saleMonth"] = data.saledate.dt.month
            data["saleDay"] =data.saledate.dt.day
            data["saleDayOfWeek"] = data.saledate.dt.dayofweek
            data["saleDayOfYear"] = data.saledate.dt.dayofyear

            data.drop("saledate", axis=1, inplace=True)


            # Fill the numeric row with median
            for label, content in data.items():
                    if pd.api.types.is_numeric_dtype(content):
                        if pd.isnull(content).sum():
                            # Add a binary column which tells us if the data was missing or not
                            data[label+"is_missing"] = pd.isnull(content)
                            # Fill missing numeric values with median
                            data[label] = content.fillna(content.median())

                    # Filled categorical missing data and turn categories into numbers
                    if not pd.api.types.is_numeric_dtype(content):
                        data[label+"is_missing"] = pd.isnull(content)
                        # We add +1 to the category code because pandas encodes missing categories as -1
                        data[label] = pd.Categorical(content).codes+1
                
        
        
            return data
        except Exception as e:
            logging.error("Error in Data handling: {}".format(e))
            raise e



class DataDivideStrategy(DataStrategy):
    """
    Data dividing strategy which divides the data into train and test data.
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divides the data into train and test data.
        """
        try:
            # Split the data into train and validation
            df_val = data[data.saleYear == 2012]
            df_train = data[data.saleYear !=2012]

            len(df_val), len(df_train)

            # Split the data into X and y
            X_train, y_train = df_train.drop("SalePrice", axis=1), df_train.SalePrice
            X_test, y_test = df_val.drop("SalePrice", axis=1), df_val.SalePrice

            X_train.shape, y_train.shape, X_test.shape, y_test.shape
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in data divide strategy: {}".format(e))
            raise e


class DataCleaning:
    """
    Data cleaning class which preprocesses the data and divides it into train and test data.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        """Initializes the DataCleaning class with a specific strategy."""
        self.df = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data based on the provided strategy"""
        return self.strategy.handle_data(self.df)