#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 11:51:14 2019

@author: pt
"""
import pandas as pd

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
from sklearn.impute import SimpleImputer

from pydispatch import Dispatcher




class GoWide(BaseEstimator, TransformerMixin):
    def __init__(self, transformer = SimpleImputer()):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """ 
        if type(transformer) == str:
            #extract transformer function from string. Grid search object gives string. 
#            try:
            transformer=eval(transformer)()
#            except KeyError:
#                raise ValueError('invalid input')
        self.transformer = transformer
            
    def fit(self, X, y=None):
        self.transformer.fit(X)
        return self
    
    def transform(self, x, y=None):        
        result = self.transformer.transform(x)
        return result
        



if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/home/pt/Documents/auto_eda')   
    from SaveAndLoad import SaveAndLoad
    from sklearn.model_selection import GridSearchCV
    from xgboost.sklearn import XGBRegressor

    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.datasets import fetch_california_housing
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.neighbors import KNeighborsRegressor
    import numpy as np


    
    database = SaveAndLoad('/media/pt/hdd/Auto EDA Results/regression/results/pickles')   
    
    combined_categorical_df = database.load('combined_categorical_df')
    combined_continuous_df = database.load('combined_continuous_df')
    y_train = database.load('y_train')
    
    joint_df = database.load('joint_df')
    
    
    
    pipeline = make_pipeline(GoWide() , XGBRegressor())

        
    
    grid_search_params = [
        {
            'gowide__transformer': [IterativeImputer()], # SVM if hinge loss / logreg if log loss
            'gowide__transformer__estimator' : [BayesianRidge()]
        },
        {
            'gowide__transformer': [IterativeImputer()], # SVM if hinge loss / logreg if log loss
            'gowide__transformer__estimator' : [DecisionTreeRegressor(max_features='sqrt', random_state=0)]
        },
         
        {
            'gowide__transformer': [IterativeImputer()], # SVM if hinge loss / logreg if log loss
            'gowide__transformer__estimator' : [ExtraTreesRegressor(n_estimators=100, max_depth = 1, random_state=0)]
        },
    
        {
            'gowide__transformer': [IterativeImputer()], # SVM if hinge loss / logreg if log loss
            'gowide__transformer__estimator' : [ExtraTreesRegressor()],
            'gowide__transformer__estimator__n_estimators' : [1],
            'gowide__transformer__estimator__n_jobs' : [-1]
        },
        
        
        {
            'gowide__transformer': [IterativeImputer()], # SVM if hinge loss / logreg if log loss
            'gowide__transformer__estimator' : [KNeighborsRegressor(n_neighbors=3)]
            
        },
    
    ]
    


    grid_search_estimator = GridSearchCV(pipeline, grid_search_params_dp, cv = 5, scoring='neg_mean_squared_log_error', n_jobs=-1, verbose = 3)

    import time
    t1 = time.time()
    grid_search_estimator.fit(joint_df, y_train)
    t2 = time.time()
    
    print('Not Nested n_jobs', t2-t1)
  
    results = grid_search_estimator.cv_results_
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#    scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
#
#
#    import matplotlib.pyplot as plt
#    
#    import numpy as np
#    from matplotlib import pyplot as plt
#    
#    from sklearn.datasets import make_hastie_10_2
#    from sklearn.model_selection import GridSearchCV
#    from sklearn.metrics import make_scorer
#    from sklearn.metrics import accuracy_score
#    from sklearn.tree import DecisionTreeClassifier
#
#
#    plt.figure(figsize=(13, 13))
#    plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
#              fontsize=16)
#    
#    plt.xlabel("min_samples_split")
#    plt.ylabel("Score")
#    
#    ax = plt.gca()
#    ax.set_xlim(0, 402)
#    ax.set_ylim(0.73, 1)
#    
#    # Get the regular numpy array from the MaskedArray
#    X_axis = np.array(results['mean_test_score'].data, dtype=float)
#    
#    for scorer, color in zip(sorted(scoring), ['g', 'k']):
#        for sample, style in (('train', '--'), ('test', '-')):
#            sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
#            sample_score_std = results['std_%s_%s' % (sample, scorer)]
#            ax.fill_between(X_axis, sample_score_mean - sample_score_std,
#                            sample_score_mean + sample_score_std,
#                            alpha=0.1 if sample == 'test' else 0, color=color)
#            ax.plot(X_axis, sample_score_mean, style, color=color,
#                    alpha=1 if sample == 'test' else 0.7,
#                    label="%s (%s)" % (scorer, sample))
#    
#        best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
#        best_score = results['mean_test_%s' % scorer][best_index]
#    
#        # Plot a dotted vertical line at the best score for that scorer marked by x
#        ax.plot([X_axis[best_index], ] * 2, [0, best_score],
#                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)
#    
#        # Annotate the best score for that scorer
#        ax.annotate("%0.2f" % best_score,
#                    (X_axis[best_index], best_score + 0.005))
#    
#    plt.legend(loc="best")
#    plt.grid(False)
#    plt.show()






#
#
#
#impute_estimators = [
#    BayesianRidge(),
#    DecisionTreeRegressor(max_features='sqrt', random_state=0),
#    ExtraTreesRegressor(n_estimators=100, max_depth = 1, random_state=0),    
#    KNeighborsRegressor(n_neighbors=4)
#]
#
#
#
## explicitly require this experimental feature
#from sklearn.experimental import enable_iterative_imputer  # noqa
## now you can import normally from sklearn.impute
#from sklearn.impute import IterativeImputer
#
#import sklearn
#
#DecisionTreeRegressor_params =      {
#                                        'max_features' : 'sqrt', 
#                                        'random_state' : 0
#                                    }
#
#
##IterativeImputer(random_state=0, estimator=impute_estimator)
#transformer_switcher_params_1 =      {
#                                        'class_name' : sklearn.impute.IterativeImputer,
#                                        'random_state' : 0,
#                                        'estimator' : BayesianRidge()
#                                   }
#
#
#transformer_switcher_params_2 =      {
#                                        'class_name' : sklearn.impute.IterativeImputer,
#                                        'random_state' : 0,
#                                        'estimator' : DecisionTreeRegressor(max_features='sqrt', random_state=0)
#                                   }
#
#
#
#
#from sklearn.pipeline import Pipeline
#
#
##steps = [('Transformer_Switcher', Transformer_Switcher(transformer_switcher_params_1)), ('XGBoost', XGBRegressor())]
##all_pipelines = Pipeline(steps)
#
#
#all_pipelines = make_pipeline(Transformer_Switcher(transformer_switcher_params_1) , XGBRegressor())
#
#
#
#all_pipelines.fit(joint_df, y_train)
#
#all_pipelines.predict(joint_df)


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











