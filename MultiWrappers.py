#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 15:03:12 2019

@author: pt
"""


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.base import RegressorMixin


class MultiTf(BaseEstimator, TransformerMixin):
    def __init__(self, transformer = SimpleImputer()):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """ 
        if type(transformer) == str:
            transformer=eval(transformer)()
        self.transformer = transformer
            
    def fit(self, X, y=None, **fit_params):
        if y is not None:
            self.transformer.fit(X)            
        return self
    
    def transform(self, x, y=None):        
        result = self.transformer.transform(x)
        return result
        





class MultiRegressor(BaseEstimator, RegressorMixin):  
    """An example of classifier"""

    def __init__(self, estimator = None):
        """
        Called when initializing the classifier
        """        
        self.estimator = estimator


    def fit(self, X, y=None):
        self.estimator.fit(X, y)
        return self


    def predict(self, X):
        return(self.estimator.predict(X))