# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from typing import Tuple

import pandas as pd
import numpy as np

class CorrelationModel:

    def __init__(self, dataset: pd.DataFrame) -> None:

        self._dataset: pd.DataFrame = dataset
        self._scaler: StandardScaler = StandardScaler()

        self._algorithm: RandomForestRegressor = \
            RandomForestRegressor(n_estimators = 1000,
                                  random_state = 42)

    def __repr__(self) -> str:
        '''Represents as string'''
        return self._algorithm
    
    def __del__(self) -> None:

        '''Deletes objects'''

        del self._dataset
        del self._algorithm
        del self._scaler
    
    @property
    def dataset(self) -> pd.DataFrame:
        '''Returns Dataset'''
        return self._dataset

    @property
    def input_series(self) -> pd.Series:
        '''Returns Scaled X Input Series as Series'''
        return self._input_series

    @input_series.setter
    def input_series(self, input_series: np.ndarray) -> None:
        '''Sets Scaled X Input Series as Series'''
        self._input_series: np.ndarray = input_series

    @property
    def scaler(self) -> StandardScaler:
        '''Returns Model Scaler'''
        return self._scaler
        
    @property
    def X(self) -> pd.DataFrame:
        '''Returns X Property As Df - Y'''
        return self.dataset.drop('Price_euros', 
                                  axis = 1)

    @property
    def y(self) -> pd.Series:
        '''Returns y Series (Price)'''
        return self.dataset['Price_euros']

    @property
    def model(self) -> object:

        '''
        Returns Trained Model
        
        Call only after training
        '''

        return self._model

    @model.setter
    def model(self, trained: object) -> None:

        '''Sets Model Property To Trained Model Instance'''

        self._model: RandomForestRegressor = trained

    def _train_test_split(self, size: float = 0.15) -> Tuple[np.ndarray]:

        '''Splits The Dataset For Training And Testing, Default size is 15%'''

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=size)

        return X_train, X_test, y_train, y_test
    
    def _scaled_test_train(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        '''Scales Test Train X Data For Forest Model Fitting'''

        X_train_scaled: np.ndarray = self.scaler.fit_transform(X_train)
        X_test_scaled: np.ndarray = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled
    
    def _fit(self, X: np.ndarray, y: np.ndarray) -> float:

        '''Forest Model Instance Fit (Train)'''

        return self._algorithm.fit(X, y)
    
    def _train_model(self) -> Tuple[np.ndarray, np.ndarray, float, float]:

        '''
        Sets Instance Model Property To Trained Model (Ready For Use)

        Returns Compiled Model Accuracy And Error Degree With Test Data
        '''

        X_train, X_test, y_train, y_test = self._train_test_split()
        X_train_scaled, X_test_scaled = self._scaled_test_train(X_train, X_test)

        self.model: float = self._fit(X_train_scaled, y_train)

        return self._calculate_accuracy(X_test_scaled, y_test)
    
    def _calculate_accuracy(self, X: np.ndarray, y: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray, float, float]:

        '''Returns Test Data With Accuracy Score And Error Degree'''

        predictions: float = self.predict(X)
        model_errors: float = abs(predictions - y)
        mape: float = 100 * (model_errors / y)
        accuracy: float = 100 - np.mean(mape)

        return (X, y, accuracy, mape)
    
    def predict(self, X: np.ndarray) -> float:

        '''Forest Model Instance Prediction'''

        return self._algorithm.predict(X)
