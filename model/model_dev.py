import logging
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


class Model(ABC):
    """
    Abstract base class for all models.
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model on the given data.

        Args:
            x_train: Training data
            y_train: Target data
        """
        pass
class LinearRegressionModel(Model):
    """
    LinearRegressionModel that implements the Model interface.
    """

    def train(self, X_train, y_train, **kwargs):
        try:
            
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info('Training complete')
            return reg
        except Exception as e:
            logging.error("Error in training model: {}".format(e))
            raise e 
        
class RandomForestRegressor(Model):
    """
    LinearRegressionModel that implements the Model interface.
    """

    def train(self, X_train, y_train, **kwargs):
        try:
            # Most ideal hyperparameters
            ideal_model = RandomForestRegressor(n_estimators=40,
                                                min_samples_leaf=1,
                                                min_samples_split=14,
                                                max_features=0.5,
                                                n_jobs=-1,
                                                max_samples=None,
                                                random_state=42)

            # Fit the ideal model
            ideal_model.fit(X_train, y_train)
            logging.info('Training complete')
            return ideal_model
        except Exception as e:
            logging.error("Error in training model: {}".format(e))
            raise e 