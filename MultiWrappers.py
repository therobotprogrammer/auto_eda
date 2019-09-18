#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 15:03:12 2019

@author: pt
"""

import gc

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.compose import TransformedTargetRegressor        

import warnings
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import check_array, safe_indexing
import numpy as np
import pandas as pd





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
        


class MultiRegressorWithTargetTransformation(BaseEstimator, RegressorMixin):  
    """An example of classifier"""

    def __init__(self, regressor = None, transformer=None, func=None, inverse_func=None, check_inverse=True):
    
        """
        Called when initializing the classifier
        """        
        self.regressor = regressor
        
        self.transformer = transformer
        self.func=None
        self.inverse_func=None
        self.check_inverse=True
        
#        if transformer is None:
#            print('Transformer is None')
        


    def _fit_transformer(self, y):
        """Check transformer and fit transformer.
        Create the default transformer, fit it and make additional inverse
        check on a subset (optional).
        """
        if (self.transformer is not None and
                (self.func is not None or self.inverse_func is not None)):
            raise ValueError("'transformer' and functions 'func'/"
                             "'inverse_func' cannot both be set.")
        elif self.transformer is not None:
            self.transformer_ = clone(self.transformer)
            
        else:
            if self.func is not None and self.inverse_func is None:
                raise ValueError("When 'func' is provided, 'inverse_func' must"
                                 " also be provided")
            self.transformer_ = FunctionTransformer(
                func=self.func, inverse_func=self.inverse_func, validate=True,
                check_inverse=self.check_inverse)
        # XXX: sample_weight is not currently passed to the
        # transformer. However, if transformer starts using sample_weight, the
        # code should be modified accordingly. At the time to consider the
        # sample_prop feature, it is also a good use case to be considered.
        self.transformer_.fit(y)
        if self.check_inverse:
            idx_selected = slice(None, None, max(1, y.shape[0] // 10))
            y_sel = safe_indexing(y, idx_selected)
            y_sel_t = self.transformer_.transform(y_sel)
            if not np.allclose(y_sel,
                               self.transformer_.inverse_transform(y_sel_t)):
                warnings.warn("The provided functions or transformer are"
                              " not strictly inverse of each other. If"
                              " you are sure you want to proceed regardless"
                              ", set 'check_inverse=False'", UserWarning)
                
                
                
    def fit(self, X, y, sample_weight=None):
#    def fit(self,*args, **kwargs):
#        self.transformer = transformer
        
#        if self.transformer is None:
#            print('Transformer is None in fit also')
#        else:
#            print('Transformer:',self.transformer)
#
#        if y is None:
#            print('y is None ')
#        else:
#            print('y is not none')
#        
#        if X is None:
#            print('x is None ')
#        else:
#            print('x is not none')
#            
#            
#        if np.isnan(X).any():
#            print('X is nan')
#        else:
#            print('X is complete')
#
#        if np.isnan(y).any():
#            print('y is nan')
#        else:
#            print('y is complete')
        

        
        """Fit the model according to the given training data.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target values.
        sample_weight : array-like, shape (n_samples,) optional
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.
        Returns
        -------
        self : object
        """
        
        #since y is series, we have to first convert it to numpy array as this function is designed for numpy array
        #this is needed for y.reshape
#        print(type(y))
        if type(y) == type(pd.Series()):
            y = y.to_numpy()
#            print(type(y))

            
#        y = check_array(y, accept_sparse=False, force_all_finite=False, ensure_2d=False, dtype='numeric')

        # store the number of dimension of the target to predict an array of
        # similar shape at predict
        self._training_dim = y.ndim

        # transformers are designed to modify X which is 2d dimensional, we
        # need to modify y accordingly.
        if y.ndim == 1:
            y_2d = y.reshape(-1, 1)
        else:
            y_2d = y
        self._fit_transformer(y_2d)

        # transform y and convert back to 1d array if needed
        y_trans = self.transformer_.transform(y_2d)
        # FIXME: a FunctionTransformer can return a 1D array even when validate
        # is set to True. Therefore, we need to check the number of dimension
        # first.
        if y_trans.ndim == 2 and y_trans.shape[1] == 1:
            y_trans = y_trans.squeeze(axis=1)


        self.regressor_ = clone(self.regressor)
        
        if sample_weight is None:
            self.regressor_.fit(X, y_trans)
        else:
            self.regressor_.fit(X, y_trans, sample_weight=sample_weight)


        #forcing garbage collection
        gc.collect()
        len(gc.get_objects()) # particularly this part!
        
        
        return self
    


    def predict(self, X):
        check_is_fitted(self, "regressor_")
        pred = self.regressor_.predict(X)
        if pred.ndim == 1:
            pred_trans = self.transformer_.inverse_transform(
                pred.reshape(-1, 1))
        else:
            pred_trans = self.transformer_.inverse_transform(pred)
        if (self._training_dim == 1 and
                pred_trans.ndim == 2 and pred_trans.shape[1] == 1):
            pred_trans = pred_trans.squeeze(axis=1)


        return pred_trans


 
    def _more_tags(self):
        return {'poor_score': True, 'no_validation': True}
    
    
    

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
        self.target_regressor = TransformedTargetRegressor(regressor=regressor, transformer=transformer, func=func, inverse_func=inverse_func, check_inverse=check_inverse)


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
    
    
    
    
    
    
    
    
    
    
    
    

        
        
        
        
        
        
        
        
        