#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 11:51:14 2019

@author: pt
"""
import itertools
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

#
#
#a =     [
#            BayesianRidge(),
#            DecisionTreeRegressor(max_features='sqrt', random_state=0),
#            ExtraTreesRegressor(n_estimators=100, max_depth = 1, random_state=0),    
#            KNeighborsRegressor(n_neighbors=4)
#        ]
#
#
#b = ['1', '2', '3', '4']
#c = ['!', '@', '#']
#
#df = pd.DataFrame(columns= ['column_a','column_b', 'column_c'])
#
#
#pipeline_combinations = []
#
#count = 10
#
#for pipeline_name in itertools.product(a, b,c):
#    pipeline_combinations.append(pipeline_name)
#
#pipeline_combinations_df = pd.DataFrame(pipeline_combinations)
#
#
# 



from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline

from sklearn import ensemble
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

from sklearn.tree import DecisionTreeRegressor
import copy



class Transformer_Switcher(BaseEstimator, TransformerMixin):
    def __init__(self, transformer = SimpleImputer()):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """ 
#        if params != {}:
#
#            extracted_estimator = params['estimator']
#    #        estimator_key.__class__.__name__
#            del params['estimator']
#            extracted_params = params
#            
#            if extracted_params == 'Default':
#                self.estimator = extracted_estimator
#            else:
#                self.estimator = extracted_estimator.set_params(**extracted_params)
        
#        
#        assert(params is not None)
#        
#        if params != {} and params is not None:
#            print('in params')
##            estimator_key = list( params.keys() )[0]
#            
#            
##            extracted_params = params[estimator_key]
#            params_local = copy.deepcopy(params)
#
#            extracted_estimator_class_name = params_local['class_name']
##            
#            del params_local['class_name']
#            self.estimator = extracted_estimator_class_name(**params_local)
#            print('Initialised')
#            print()
#            
#        else:
#            return
        
        self.transformer = transformer
            
    def fit(self, X, y=None):
        self.transformer.fit(X)
        return self
    

    def transform(self, x, y=None):
        
        result = self.transformer.transform(x)
        return result
        





pipeline = make_pipeline(Transformer_Switcher() , XGBRegressor())


transformer_switcher_params_1 =      {
                                        'class_name' : sklearn.impute.IterativeImputer,
                                        'random_state' : 0,
                                        'estimator' : BayesianRidge()
                                   }




parameters = [
    {
        'transformer_switcher__transformer': [sklearn.impute.IterativeImputer()], # SVM if hinge loss / logreg if log loss
        'transformer_switcher__transformer__estimator' : [BayesianRidge()],
    },
    {
        'transformer_switcher__transformer': [sklearn.impute.IterativeImputer()], # SVM if hinge loss / logreg if log loss
        'transformer_switcher__transformer__estimator' : [DecisionTreeRegressor(max_features='sqrt', random_state=0)],
    },
     
    {
        'transformer_switcher__transformer': [sklearn.impute.IterativeImputer()], # SVM if hinge loss / logreg if log loss
        'transformer_switcher__transformer__estimator' : [ExtraTreesRegressor(n_estimators=100, max_depth = 1, random_state=0)]
    },

    {
        'transformer_switcher__transformer': [sklearn.impute.IterativeImputer()], # SVM if hinge loss / logreg if log loss
        'transformer_switcher__transformer__estimator' : [KNeighborsRegressor(n_neighbors=4)],
        
    },

]




gscv = GridSearchCV(pipeline, parameters, cv=10, n_jobs=-1, verbose=3, scoring = global_scoring)
# param optimization
gscv.fit(joint_df, y_train)

t = gscv.cv_results_


estimator.get_params().keys()













impute_estimators = [
    BayesianRidge(),
    DecisionTreeRegressor(max_features='sqrt', random_state=0),
    ExtraTreesRegressor(n_estimators=100, max_depth = 1, random_state=0),    
    KNeighborsRegressor(n_neighbors=4)
]



# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer

import sklearn

DecisionTreeRegressor_params =      {
                                        'max_features' : 'sqrt', 
                                        'random_state' : 0
                                    }


#IterativeImputer(random_state=0, estimator=impute_estimator)
transformer_switcher_params_1 =      {
                                        'class_name' : sklearn.impute.IterativeImputer,
                                        'random_state' : 0,
                                        'estimator' : BayesianRidge()
                                   }


transformer_switcher_params_2 =      {
                                        'class_name' : sklearn.impute.IterativeImputer,
                                        'random_state' : 0,
                                        'estimator' : DecisionTreeRegressor(max_features='sqrt', random_state=0)
                                   }




from sklearn.pipeline import Pipeline


#steps = [('Transformer_Switcher', Transformer_Switcher(transformer_switcher_params_1)), ('XGBoost', XGBRegressor())]
#all_pipelines = Pipeline(steps)


all_pipelines = make_pipeline(Transformer_Switcher(transformer_switcher_params_1) , XGBRegressor())



all_pipelines.fit(joint_df, y_train)

all_pipelines.predict(joint_df)


#
#all_pipelines = make_pipeline(Transformer_Switcher(transformer_switcher_params_1) , XGBRegressor())
#
#grid_params =   { 
#                    'transformer_switcher__params': [transformer_switcher_params_1]
#
#                }
#
#
#
#CV = GridSearchCV(estimator = all_pipelines, param_grid = grid_params,  n_jobs= 1)
#
#
#
#
#CV.fit(joint_df.values, y_train.values)
#
#
#











