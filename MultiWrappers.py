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
from sklearn.preprocessing import FunctionTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted



global_garbage_collection = False

class MultiTf(BaseEstimator, TransformerMixin):
    def __init__(self, transformer= SimpleImputer()):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """ 
#        if type(transformer) == str:
#            transformer=eval(transformer)()
        self.transformer = transformer
        
#        print('transformer is: ', type(self.transformer).__name__)

            
    def fit(self, X, y=None):
#        assert y is not None
        self.transformer = self.transformer
        
#        print('transformer is: ', type(self.transformer).__name__)
        self.transformer.fit(X, y=None)  
        
        #forcing garbage collection
        if global_garbage_collection:
            gc.collect()
            len(gc.get_objects()) # particularly this part!
            
        
        return self
       
    def fit_transform(self, X, y=None):       
        return self.transformer.fit_transform(X, y=None)
        
    def transform(self, X):      
#        print('transformer is: ', type(self.transformer).__name__)
        X = self.transformer.transform(X)
        return X
    
    def _more_tags(self):
        return self.transformer._more_tags


class MultiFunctionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, func=None, inverse_func=None, validate=None, accept_sparse=False, pass_y='deprecated', check_inverse=True, kw_args=None, inv_kw_args=None, name = ''):       
        self.transformer = FunctionTransformer(func=None, inverse_func=None, validate=None, accept_sparse=False, pass_y='deprecated', check_inverse=True, kw_args=None, inv_kw_args=None)
        self.name = name

    def _check_input(self, X):
        return self.transformer._check_input(X)

    def _check_inverse_transform(self, X):
        self.transformer._check_inverse_transform(self, X)
        
        
        

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
        



class NamedFunctionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, func, inverse_func, name, validate=None,
             accept_sparse=False, pass_y='deprecated', check_inverse=True,
             kw_args=None, inv_kw_args=None):
        self.func = func
        self.inverse_func = inverse_func
        self.validate = validate
        self.accept_sparse = accept_sparse
        self.pass_y = pass_y
        self.check_inverse = check_inverse
        self.kw_args = kw_args
        self.inv_kw_args = inv_kw_args
        self._name = name
        
        self.function_transformer = FunctionTransformer(func=None, inverse_func=None, validate=None,
             accept_sparse=False, pass_y='deprecated', check_inverse=True,
             kw_args=None, inv_kw_args=None)
        

    def _check_input(self, X):
        return self.function_transformer._check_input(X)
    
    
    def _check_inverse_transform(self, X):
        self.function_transformer._check_inverse_transform(X)
        
    def fit(self, X, y=None):
        self.function_transformer.fit(X, y=None)
        return self
    
    def transform(self, X):
        return self.function_transformer.transform(X)
    
    def inverse_transform(self, X):
        return self.function_transformer.inverse_transform(X)
    
    def _transform(self, X, func=None, kw_args=None):
        return self.function_transformer._transform(X, func=None, kw_args=None)
    
    def _more_tags(self):
        return self.function_transformer._more_tags()
    

class MultiRegressorWithTargetTransformation(BaseEstimator, RegressorMixin):  
    """An example of classifier"""

    def __init__(self, regressor = None, transformer=None, func_inverse_func_pair = None, check_inverse=True):
    
        """
        Called when initializing the classifier
        """        
        self.regressor = regressor
        
        self.transformer = transformer
        self.func_inverse_func_pair = func_inverse_func_pair
        self.check_inverse=True
        
        
        if self.func_inverse_func_pair is not None:
            self.func = self.func_inverse_func_pair[0]
            self.inverse_func= self.func_inverse_func_pair[1]
        else:
            self.func = None
            self.inverse_func= None        
        
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
        
        if self.func_inverse_func_pair is not None:
            self.func = self.func_inverse_func_pair[0]
            self.inverse_func= self.func_inverse_func_pair[1]
        
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
        if global_garbage_collection:
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
        if global_garbage_collection:
            gc.collect()
            len(gc.get_objects()) # particularly this part!
        return self
    
    def fit(self, X, y, sample_weight=None):
        self.target_regressor.fit(X, y, sample_weight=None)
        
        #forcing garbage collection
        if global_garbage_collection:
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
    
    
    
    
    
    
    
    
    
    
    
    

        
        
        
        
        
        
        
        
        