#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 15:03:12 2019

@author: pt
"""

import gc

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
            
            #forcing garbage collection
            gc.collect()
            len(gc.get_objects()) # particularly this part!
            
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
        
        #forcing garbage collection
        gc.collect()
        len(gc.get_objects()) # particularly this part!
        return self


    def predict(self, X):
        return(self.estimator.predict(X))
        
        
        

from sklearn.compose import TransformedTargetRegressor        

class MultiTransformedTargetRegressor(BaseEstimator, RegressorMixin):  
    """An example of classifier"""

    def __init__(self, regressor=None, transformer=None, func=None, inverse_func=None, check_inverse=True):
        """
        Called when initializing the classifier
        """ 
        
        self.regressor = regressor
        self.transformer = transformer
        self.func = func
        self.inverse_func = inverse_func
        self.check_inverse = check_inverse
        self.target_regressor = TransformedTargetRegressor(regressor=None, transformer=None, func=None, inverse_func=None, check_inverse=True)


    def _fit_transformer(self, y=None):
        self.target_regressor._fit_transformer(y)
        
        #forcing garbage collection
        gc.collect()
        len(gc.get_objects()) # particularly this part!
        return self
    
    def fit(self, X, y, sample_weight=None):
        self.target_regressor.fit(X, y, sample_weight=None)
        
        #forcing garbage collection
        gc.collect()
        len(gc.get_objects()) # particularly this part!
        return self


    def predict(self, X):        
        return self.target_regressor.predict(X)      
        
    def _more_tags(self):
        return self.target_regressor._more_tags()
        


if __name__ == '__main__':
        
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.preprocessing import PowerTransformer

    
    tt = MultiTransformedTargetRegressor(regressor=LinearRegression(),func=np.log, inverse_func=np.exp)
    X = np.arange(40).reshape(-1, 1)
    y = np.exp(2 * X).ravel()
    tt.fit(X, y) 
    
    tt.score(X, y)
    
    
    from sklearn import svm, datasets
    from sklearn.model_selection import GridSearchCV
    iris = datasets.load_iris()
    
    parameters = {'transformer': [QuantileTransformer, PowerTransformer]}
    svc = MultiTransformedTargetRegressor(regressor=LinearRegression())
    clf = GridSearchCV(svc, parameters, cv=5)
    clf.fit(iris.data, iris.target)
    
    clf.score(iris.data, iris.target)
    
    
    
    
    
    
    
    
    
    
    
    

        
        
        
        
        
        
        
        
        