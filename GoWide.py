#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 11:51:14 2019

@author: pt
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer

import itertools
import numpy as np

from Plotter import Plotter
import numbers
    



def gererate_params(config_dict):
    generated_params = __dp(config_dict)
    return generated_params


def __dp(curr_item = None, prefix = '', depth = 0):
    if type(curr_item) == dict:
        params = []
        for key, value in curr_item.items():            
            if type(key) == str:
                if prefix == '':
                    res = __dp(curr_item = value, prefix = key, depth = depth+1)
                    
                else:
                    res = __dp(curr_item = value, prefix = prefix + '__'+ key, depth = depth+1)
            else:
                #key is assumed to be a function
                estimator_key_value = {prefix: [key]}
                res = __dp(curr_item = value, prefix = prefix , depth = depth+1)
                
                for sub_dict in res:
                    sub_dict.update(estimator_key_value)
            params.append(res)
            
        permutations = []
        
        for element in itertools.product(*params):           
            d = {}     
            for dict_value in element:
                d.update(dict_value)            
            permutations.append(d)
        return permutations        
                
    elif type(curr_item) == list:
        params = []        
        for value in curr_item:                
#            if type(value) == tuple:
#                res = __dp(curr_item = value[1], prefix = prefix + '__'+ value[0], depth = depth+1)
#            else:                
            res = __dp(value, prefix , depth = depth+1)
            if type(res) == list:
                for item in res:
                    params.append(item)
            else:
                params.append(res)
        return params
        
    else:
        return {prefix: [curr_item]}



   
def __get_params_and_scores(results):        
    parameters_to_plot = pd.DataFrame(results['params'])
    scores = pd.DataFrame()

    for key, value in results.items():
        if type(results[key] ) == np.ndarray:
            scores[key] = value

    processed_results_dict = {}
    processed_results_dict['parameters_to_plot'] = parameters_to_plot
    processed_results_dict['scores'] = scores
    
    return processed_results_dict


def __convert_functions_to_class_names(processed_results_dict):
    df_local = processed_results_dict['parameters_to_plot'].copy()
#    locations_of_functions_mask = pd.DataFrame(np.zeros((df_local.shape[0], df_local.shape[0])))

#    locations_of_functions_mask = pd.DataFrame(0, index=range(df_local.shape[0]), columns=range(df_local.shape[1])))

    total_rows = df_local.shape[0]
    total_cols = df_local.shape[1]
    function_locations_mask = pd.DataFrame(False, index=range(total_rows), columns=df_local.columns)


    for column_idx, column in enumerate(df_local.columns):
        for row_idx, value in enumerate(df_local[column]) :
            cell_value = df_local.iloc[row_idx, column_idx]
            if not (isinstance(cell_value, str) or isinstance(cell_value, numbers.Number) ):
                try:
                    df_local.iloc[row_idx, column_idx] = type(cell_value).__name__
                    function_locations_mask.iloc[row_idx, column_idx] = True
                    
                except:
                    df_local.iloc[row_idx, column_idx] = str(cell_value)
                    
    processed_results_dict['parameters_to_plot'] = df_local
    processed_results_dict['functions_to_plot'] = df_local.where(function_locations_mask, np.nan)   
                
    return processed_results_dict


def __remove_variance(processed_results_dict, compact_version = False):
    temp_df = processed_results_dict['parameters_to_plot'].copy()        
#    temp_df = temp_df.dropna(axis = 1, how = 'all')  

    if not compact_version:
        temp_df = temp_df.fillna('##MISSING##')
          
    columns_with_no_varience = []        
    
    for column in temp_df.columns:
        group_obj =  temp_df.groupby(column, observed = True)            
        group_count_in_column = group_obj.count().shape[0]

        if group_count_in_column < 2:
            columns_with_no_varience.append(column)   
            
    
    processed_results_dict['parameters_to_plot'] = processed_results_dict['parameters_to_plot'].drop(columns_with_no_varience, axis = 1)     
    processed_results_dict['functions_to_plot'] = processed_results_dict['functions_to_plot'].drop(columns_with_no_varience, axis = 1)     
    return processed_results_dict    


def process_results(cv_results, compact_version = False):
    processed_results_dict = __get_params_and_scores(cv_results)
    processed_results_dict =  __convert_functions_to_class_names(processed_results_dict)
    processed_results_dict = __remove_variance(processed_results_dict, compact_version)
    

    return processed_results_dict



import re, mpu
def split_at_char_int_transistion(text_str):                       
    if (text_str == np.nan) or (text_str == None) or (text_str == 'nan') or type(text_str) == type(np.nan) :
       return np.nan
   
    elif not type(text_str) == type([]):
        temp = []
        temp.append(text_str)
        text_str = temp
        
    elif type(text_str) == type([]):
        text_str = mpu.datastructures.flatten(text_str)
    
    new_list = []

    for element in text_str:
        
#        filtered_element = element.translate(str.maketrans(filters, ' ' * len(filters)))    
#        split_list = re.split('(\d+)',element)
        split_list = re.findall('(\d+|\D+)', element)
        new_list.append(split_list)
    
    new_list = mpu.datastructures.flatten(new_list) 
    
    return new_list


    


class MultiTf(BaseEstimator, TransformerMixin):
    def __init__(self, transformer = SimpleImputer()):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """ 
        if type(transformer) == str:
            transformer=eval(transformer)()
        self.transformer = transformer
            
    def fit(self, X, y=None):
        self.transformer.fit(X)
        return self
    
    def transform(self, x, y=None):        
        result = self.transformer.transform(x)
        return result
        

                

if __name__ == '__main__':
    from mlflow import log_metric, log_param, log_artifacts


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
    np.random.seed(0)

    #reimport because if an issue where pickle cannot find a class not part of __main__
    from GoWide import MultiTf
    import time



    database = SaveAndLoad('/media/pt/hdd/Auto EDA Results/regression/results/pickles')   
    
    combined_categorical_df = database.load('combined_categorical_df')
    combined_continuous_df = database.load('combined_continuous_df')
    y_train = database.load('y_train')
    
    joint_df = database.load('joint_df')
    
    
    memory = '/media/pt/hdd/Auto EDA Results/regression/results/memory'


    ExtraTreesRegressor_params =    {
                                        'max_depth': [1, 2, 3], 
                                        'n_estimators': [1,10,100,1000],
                                    }
    
    
    KNeighborsRegressor_params =    {
                                        'n_neighbors' : [2],
                                    }
    
    
    estimator_list =                [   
                                        BayesianRidge(),
                                        DecisionTreeRegressor(),
                                        {KNeighborsRegressor() : KNeighborsRegressor_params},
                                        {ExtraTreesRegressor() : ExtraTreesRegressor_params}
                                    ]
    
    
    iterative_imputer_params =      {
                                        'estimator' : estimator_list,
                                        'missing_values' : [np.nan],
                                    }
    
    
    iterative_imputer_dict =        {
                                        IterativeImputer() : iterative_imputer_params
                                    }
    
    
    simple_imputer_params =         {
                                        'strategy' : ['mean', 'median', 'most_frequent']
                                    }
    
    
    simple_imputer_dict =           {
                                        SimpleImputer() : simple_imputer_params
                                    }
    
    
    multitf_params =                {
                                        'transformer' : [iterative_imputer_dict, simple_imputer_dict]
                                    }
    
    
    config_dict =                   {   
                                        'multitf' : multitf_params
                                    }    
    
    
    steps = [
                ('multitf' , MultiTf() ), 
                ('xgb' , XGBRegressor() ) 
            ]
    
    
    pipeline = Pipeline( memory = memory, steps = steps)
    grid_search_params = gererate_params(config_dict)
    
    grid_search_estimator = GridSearchCV(pipeline, grid_search_params, cv = 10, scoring='neg_mean_squared_log_error', n_jobs=-1, verbose = 0)
    t1 = time.time()
    grid_search_estimator.fit(joint_df, y_train)
    t2 = time.time()    
    print('Time Taken:', t2-t1)
  
    cv_results = grid_search_estimator.cv_results_

    
    processed_results_dict= process_results(cv_results, compact_version = False)

    plot_dir = '/media/pt/hdd/Auto EDA Results/unit_tests/plots'
    plotter = Plotter(plot_dir)        
    plotter.parallel_categories(processed_results_dict['parameters_to_plot'] , processed_results_dict['scores'], score_name_to_plot = 'mean_test_score')

    
    lower_is_better = False
    
    
    def sub_plots(functions_to_plot):
        for column in functions_to_plot.columns:            
            group_obj =  functions_to_plot.groupby(column)     
            group_dict = group_obj.groups
            group_count_in_column = group_obj.count().shape[0]        

            if group_count_in_column > 0:
                
                print('plotting: ', column)
                df_to_plot = pd.DataFrame()

                for key, value in group_dict.items():
                    selected_scores = processed_results_dict['scores'].loc[value, ['mean_test_score']]
                    if lower_is_better:
                        best_score_row = selected_scores.idxmin()
                    else:
                        best_score_row = selected_scores.idxmax()
                                      
                    extracted_df = processed_results_dict['scores'].loc[best_score_row].copy()
                    
                    extracted_df = extracted_df.T
                    extracted_df.columns = [key]
                    df_to_plot = df_to_plot.join(extracted_df, how = 'outer')
                    



                def get_split_scores_df(df):
                    df_local = df.copy()
                    
                    index_to_drop = []
                    
                   
                    
                    for idx, value in enumerate(df.index):
                        score_name = split_at_char_int_transistion(value)[0]
                        
                        if not score_name == 'split':
                            index_to_drop.append(value)                        
                    
                    df_local = df_local.drop(index_to_drop)
                    
                    return df_local
                
                split_scores_df = get_split_scores_df(df_to_plot)
                    
                    
                plotter.box_plot_df(split_scores_df)
                        
    sub_plots(processed_results_dict['functions_to_plot'])
        
            
    
    

    
    
    

#        
#    
#    grid_search_params = [
#        {
#            'gowide__transformer': [IterativeImputer()], # SVM if hinge loss / logreg if log loss
#            'gowide__transformer__estimator' : [BayesianRidge()]
#        },
#        {
#            'gowide__transformer': [IterativeImputer()], # SVM if hinge loss / logreg if log loss
#            'gowide__transformer__estimator' : [DecisionTreeRegressor(max_features='sqrt', random_state=0)]
#        },
#         
#        {
#            'gowide__transformer': [IterativeImputer()], # SVM if hinge loss / logreg if log loss
#            'gowide__transformer__estimator' : [ExtraTreesRegressor(n_estimators=100, max_depth = 1, random_state=0)]
#        },
#    
#        {
#            'gowide__transformer': [IterativeImputer()], # SVM if hinge loss / logreg if log loss
#            'gowide__transformer__estimator' : [ExtraTreesRegressor()],
#            'gowide__transformer__estimator__n_estimators' : [1],
#            'gowide__transformer__estimator__n_jobs' : [-1]
#        },
#        
#        
#        {
#            'gowide__transformer': [IterativeImputer()], # SVM if hinge loss / logreg if log loss
#            'gowide__transformer__estimator' : [KNeighborsRegressor(n_neighbors=3)]
#            
#        },
#    
#    ]
#    
#

    
    
    
    
    
    
    
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







