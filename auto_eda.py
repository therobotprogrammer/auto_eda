#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 16:37:03 2019

@author: pt
"""


# To Do: maek 

##############################################
# ALL TO DO
# try without using missing data
# to do: change hash function as md5 data to be hashed randomly. As a result the rf imputer throws it. 
# to do: change hash function as md5 data to be hashed randomly. As a result the rf imputer throws it.  


from ParallelCPU import ParallelCPU

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing, neighbors, decomposition, manifold, metrics
from sklearn.preprocessing import LabelEncoder
import keras
import math
import mpu
import string
import re
import sys
from sklearn_pandas import CategoricalImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn import tree, model_selection, ensemble
from sklearn.externals.six import StringIO 
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
import io
import os
import pydot
from sklearn.manifold import Isomap
from sklearn import ensemble
from sklearn import feature_selection
from scipy import stats

from concurrent.futures import ProcessPoolExecutor

import pandas_profiling
import webbrowser, os

import xgboost as xgb




from missingpy import KNNImputer, MissForest
import multiprocessing


import plotly
import plotly.graph_objs as go


import cufflinks as cf
cf.go_offline()


import plotly.io as pio
pio.renderers
pio.renderers.default = "browser"







global_problem_type = 'regression'
auto_generated_data_df_dropped = pd.DataFrame()


def log_rmse(y_orig, y_pred):
    return math.sqrt(metrics.mean_squared_log_error(y_orig,y_pred) )

def drop_and_log_column(df, column_name = None, reason = None):    
    auto_generated_data_df_dropped[column_name + '__' + reason] = df[column_name].copy(deep=True)
    df.drop(columns = column_name, inplace = True)
    

def drop_and_log_column_at_index(df, index = None, reason = None): 
    column_name = df.columns[index]
    auto_generated_data_df_dropped[column_name + '__' + reason] = df[column_name].copy(deep=True)
    df.drop(index = column_name, inplace = True)
    

    

if global_problem_type == 'categorical':
    directory = '/media/pt/hdd/Auto EDA Results/categorical' 
    file_path = os.path.join(directory, 'train.csv')
    train = pd.read_csv(file_path, index_col = False)
    

    file_path = os.path.join(directory, 'pass_nationalities.csv')
    name_prism = pd.read_csv(file_path, index_col = False)
    name_prism_train = name_prism[:train.shape[0]]
    train['nationality'] = name_prism_train['Nationality']    
    
    target_column = 'Survived'
    
    exclude_from_ohe = ['Pclass', 'SibSp', 'Parch' ]
#    train.drop[]
    
    
else:
    directory = '/media/pt/hdd/Auto EDA Results/regression'   
    
    file_path = os.path.join(directory, 'train.csv')
    train = pd.read_csv(file_path, index_col = False)
    target_column = 'SalePrice'
#    global_scoring = metrics.make_scorer(log_rmse, greater_is_better=False)
    global_scoring = 'neg_mean_squared_log_error'
    exclude_from_ohe = None

    
    


if global_problem_type == 'classification':
    drop_and_log_column(train, 'PassengerId', 'manually dropped - unique identifier')
elif global_problem_type == 'regression':
    df = drop_and_log_column(train, 'Id', 'manually dropped - unique identifier')
    

   

results_dir = os.path.join(directory + '/results')  
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)


y_train = train[target_column].copy(deep = True)  
target_df = train[target_column].copy(deep = True)

y_train_dict = {}
y_train_dict['original'] = y_train





global_log_warnings = set()
show_plots = False

global_cores = multiprocessing.cpu_count()      
global_verify_parallel_execution = False
global_debug_mode = False



parallel = ParallelCPU(verify_parallel_execution = global_verify_parallel_execution, debug_mode = global_debug_mode)



train.info()
train.describe()

max_groups_in_categorical_data = 50
min_members_in_a_group = 5

auto_min_entries_in_continous_column = 10
hashing_dimention = 2
min_data_in_column_percent = 0 #percent of nan . To Do: This should also be applied to continuous
hashing_strategy = 'md5'


class estimator:
    name = ''
    estimator = None
    X_fit_transformed = pd.DataFrame()    
    best_estimator = None
    traces = None   
    best_score = None
    best_params = None  


class dataset:    
    history = []
    X_train = pd.DataFrame()
    
estimators_dict = {}
X_train_dict = {}

use_cuda_tsne = True
if use_cuda_tsne:    
    from tsnecuda import TSNE




#to do: hash char as a to z and int as a range
#to do: for things that are hashed, use box plot etc




def is_valid(text_str):
    if (text_str == np.nan) or (text_str == None) or (text_str == 'nan') or (text_str == ''):
        return False
    else:
        return True

def replace_incomplete_data_with_missing(data):
    if np.isnan(data):
        return 'missing'
    else:
        return data
    
    
    
def clean_and_split_text(text_str, split_priority_1 = ' ', split_priority_2 = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):
    #split by space   
    if (text_str == np.nan) or (text_str == None) or (text_str == 'nan') or type(text_str) == type(np.nan) :
       return np.nan
   
    level_1_splits = keras.preprocessing.text.text_to_word_sequence(text_str, filters=split_priority_1, lower = False, split=" ")     
    
    all_level_2_splits = []    
    
    for entry in level_1_splits:
       level_2_split = keras.preprocessing.text.text_to_word_sequence(entry, filters = split_priority_2, lower = False)   
       level_2_split = ''.join(level_2_split)
       all_level_2_splits.append(level_2_split)
       
    return all_level_2_splits



def clean_text(text_str, filters = '\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):
    #split by space   
    if (text_str == np.nan) or (text_str == None) or (text_str == 'nan') or type(text_str) == type(np.nan) :
       return np.nan
    
    elif not text_str == type(list):
        temp = []
        temp.append(text_str)
        text_str = temp
        
    elif text_str == type(list):
        text_str = mpu.datastructures.flatten(text_str)
    
    new_list = []
    
    for element in text_str:
#        filtered_element = element.translate(str.maketrans(filters, ' ' * len(filters)))    
                                            
        translation_table = dict.fromkeys(map(ord, filters), None)
        filtered_element = element.translate(translation_table)                                     

        new_list.append(filtered_element)
    
    new_list = mpu.datastructures.flatten(new_list)       
    return new_list


def split_text(list_of_texts, split_at = ' '):
    new_list = []

    if (list_of_texts == np.nan) or (list_of_texts == None) or (list_of_texts == 'nan') or type(list_of_texts) == type(np.nan) :
       return np.nan
    
    for element in list_of_texts:
        new_list.append(element.split(split_at))
        
    new_list = mpu.datastructures.flatten(new_list)     
    return new_list


def split_at_char_int_transistion(text_str):
    if text_str == 'A5' or text_str == ['A5'] :
        print('Found A5')
        
    if (text_str == np.nan) or (text_str == None) or (text_str == 'nan') or type(text_str) == type(np.nan) :
       return np.nan
   
    elif not type(text_str) == type([]):
        temp = []
        temp.append(text_str)
        text_str = temp
        
    elif type(text_str) == type([]):
        text_str = mpu.datastructures.flatten(text_str)
    
    new_list = []
    if text_str == 'A5':
        print('Found A5')
    for element in text_str:
        
#        filtered_element = element.translate(str.maketrans(filters, ' ' * len(filters)))    
#        split_list = re.split('(\d+)',element)
        split_list = re.findall('(\d+|\D+)', element)
        new_list.append(split_list)
    
    new_list = mpu.datastructures.flatten(new_list) 
    
    return new_list


def hash_function_first_char(s):
    if not type(s) == str:
        s = str(s)
        warn('hash_function_first_char - converted int to string. possible loss of information')
        
    #convert to lower case
    s_lower = s.lower()
    
    return ord(s_lower[0])
    
    

def hash_it(text, strategy):
    
    if strategy == 'md5':
        hash_function = 'md5'
        n = hashing_dimention
    elif strategy == 'ascii':
        hash_function = hash_function_first_char
        n = 256
   
        
    if (text == np.nan) or (text == None) or (text == 'nan') or type(text) == type(np.nan) :
       return np.nan
    text = str(text)
    hash_list = keras.preprocessing.text.hashing_trick(text, n, hash_function = hash_function, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')

    #convert list of hashes to a single hash. This is because keras hashes multiple works together and returns their hashes 
    #as a list. But in this case we only have a single hash. So the list is collapsed and then converted to a string
    hash_str = ''.join(str(e) for e in hash_list)
    
    #convert hash string to int
    hash_int = int(hash_str)    
                                                     
    return(hash_int)   
                                              

def revert_nans_from_df(df, masks):
    assert df.shape == masks.shape
    
    rows = masks.shape[0]
    cols = masks.shape[1]
    
    for r in range(rows):
        for c in range(cols):
            if masks.iloc[r, c]:
                df.iloc[r,c] = math.nan
    return df 
        
    


def label_encoder(df_original, strategy = 'keep_missing_as_nan', offset_labels = True):    
    
    assert (min_data_in_column_percent >= 0 and min_data_in_column_percent <= 1), 'min_data_in_column_percent out of range 0 to 1'
        
    
    df = df_original.copy(deep = True)
    
    mask_df = pd.DataFrame(df.isnull().values)
    
#    mask_df.columns = df.columns
        
    sys_min = -sys.maxsize -1   
    
    
#    label_encoder_dict = {}
    
    for column_name in df.columns:  
        total_nan = combined_categorical_df[column_name].isna().sum()
        total_real_data = df.shape[0] -total_nan
        total_real_data_percentage = total_real_data / df.shape[0]
        
        if total_real_data_percentage < min_data_in_column_percent:
            auto_generated_data_df_dropped[column_name + '_failed_drop_threshold_percent_' + str(min_data_in_column_percent)] = df[column_name].copy(deep = True)
            df.drop(columns = column_name, inplace = True)            
            
        else:    
            series = df[column_name]  
           
                              
            
            if strategy == 'seperate_missing':
                
                if df[column_name] .dtype.kind in 'biufc': 
                    #convert NaN to number
                    series = series.replace(np.NaN, sys_min)
                else:
                    series = series.replace(np.NaN, 'Unknown')                          
            else:
                ci = CategoricalImputer()
                ci.fit(series)
                series = ci.transform(series)  
                
#            elif strategy == 'keep_missing_as_nan':
                
            
                
            # label encode. label encoder does not work directly on multiple columns so its in for loop    
            le = preprocessing.LabelEncoder()
            le.fit(series)
            series = le.transform(series)        
            df[column_name] = series 
            
            #this is done because later corelation heatmap is drawn. if there is only one feature and rest are nan, than 
            #that one feature gets a label of 0. Then sklearn has a bug that gives no corelation in heatmap with only 0 and nan values
            if offset_labels:
                df[column_name]  = df[column_name]  + 1
#            
#            label_encoder_dict['column_name'] = le
#    ohe = preprocessing.OneHotEncoder(categories = 'auto')
#    ohe.fit(df)
#    df = ohe.transform(df).toarray()
#    
#    df = pd.get_dummies(df)
        
    #one hot encoding   




    
    if strategy == 'keep_missing_as_nan':
        df = revert_nans_from_df(df, mask_df)

    
        
#    processed_df = df_excluded_from_one_hot.join( df_one_hot, how = 'outer')

        
    

    return df

def warn(w):       
    global_log_warnings.add(w)
    print(w)
    
    




def cont_preprocess(df_original, missing_data_drop_threshold = None, imputer = 'SimpleImputer', si_strategy = 'mean', si_fill_value = np.finfo(np.float64).min):
    
    df = df_original.copy(deep = True)

    all_columns = df.columns
    
    for column_name in all_columns:    
        series = df[column_name]
        log_column_str = '[' + column_name +']'

        
        if missing_data_drop_threshold == None and series.count() < .5 * y_train.shape[0]:
            print('WARNING: ' , log_column_str, ' Continuous column has less than drop threshold' ,  missing_data_drop_threshold, 'Try increasing auto_min_entries_in_continous_column so it can be considered caregorical')

        elif missing_data_drop_threshold != None and series.count() < missing_data_drop_threshold * y_train.shape[0]:            
            print('WARNING: ' , log_column_str, ' DROPPED: Continuous column has less than drop threshold' ,  missing_data_drop_threshold, 'Try increasing auto_min_entries_in_continous_column so it can be considered caregorical')
            
            drop_and_log_column(df, column_name, '< missing_data_threshold. Cont')
            
    
    features = df.columns
    
    if imputer == 'SimpleImputer':

#            if si_strategy == 'constant':
        imp = SimpleImputer(strategy = si_strategy, missing_values = np.nan, fill_value = si_fill_value)    
#            elif 
#                si_strategy == 'mean'
        imp.fit(df[features])
        df[features] = imp.transform(df[features])    

    return df




def analyze_tree(dt, X_train, y_train, max_depth_search_list_length = 10, filename = 'tree.pdf'):
    dot_data = StringIO()
    
    os.getcwd()
    file_path = os.path.join(results_dir, filename)
    
    feature_names = list(X_train.columns)
    feature_names = feature_names[0:10]
    
    if global_problem_type == 'classification':
        class_names = y_train.name
    else:
        class_names = None
            
    export_graphviz(dt, out_file=dot_data,  
                    filled=True, rounded=True,
                    feature_names = list(X_train.columns), class_names = class_names)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
     
    #Image(graph.create_png())
    
    graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
    graph.write_pdf(file_path)



    feature_importance_df = pd.DataFrame()
    feature_importance_df['Name'] = X_train.columns
    feature_importance_df['Importance'] = dt.feature_importances_
    
    max_depth = dt.tree_.max_depth

    #this works even if max_depth is less than max_depth_search_list_length
#    max_depth_search_list = np.linspace(1,max_depth,max_depth_search_list_length)
#    max_depth_search_list = max_depth_search_list.astype(int)
#    max_depth_search_list = np.unique(max_depth_search_list)
#    max_depth_search_list = max_depth_search_list.tolist()
    
    max_depth_search_list = get_equally_spaced_numbers_in_range(1, max_depth, max_depth_search_list_length )
    
    #more evenly spaced numbers but gives extra values
    
#    increment = (max_depth-1) //max_depth_search_list_length
#    if increment == 0:
#        increment = 1
#        
#    max_depth_search_list = list(range(1,max_depth,increment))
#    max_depth_search_list.append(max_depth)
#    
    
    results =   {
                    'feature_importance': feature_importance_df,
                    'max_depth_search_list': max_depth_search_list
                }
    return results




def get_equally_spaced_numbers_in_range(min_num = 1, max_num =100, total_numbers =10):
    equally_spaced_numbers_list = np.linspace(min_num, max_num,total_numbers)
    equally_spaced_numbers_list = equally_spaced_numbers_list.astype(int)
    equally_spaced_numbers_list = np.unique(equally_spaced_numbers_list)
    equally_spaced_numbers_list = equally_spaced_numbers_list.tolist()
    return equally_spaced_numbers_list



def get_equally_spaced_non_zero_floats_in_range(start = 0, stop =1, total_numbers = 10):        
    step = (stop-start) / total_numbers
    
    if start == 0:
        start = step
    return(np.arange(start, stop + step, step).tolist())




def plot_dataframe(df_local, message = ''):   
#    cont_column_names = list(combined_continuous_df.columns)    
    
    df_local_continuous_only = df_local.select_dtypes(exclude = 'category')  
    
    ax = df_local.plot.kde()
    ax.set_title(message)
    ax.legend(ncol = 7, prop={'size': 9})

#    df_local_continuous_only = df_local.reindex(columns = cont_column_names)  
    
    
    df_local_categorical_only = df_local.select_dtypes(include = 'category')    
#    df_local_categorical_only = df_local.reindex(columns = cont_column_names)  
    
    
#    ax = df_local.plot.kde()
#    ax.set_title(message)
#    ax.legend(ncol = 7, prop={'size': 9})
#
#    ax = df_local.plot.hist()
#    ax.set_title(message)
#    ax.legend(ncol = 7, prop={'size': 9})

#    df_local_categorical_only.iplot(kind='box', boxpoints='outliers', title = message + 'Box Plot - Categorical Data')

    df_local_continuous_only.iplot(kind='box', boxpoints='outliers', title = message + 'Box Plot - Continuous Data')

    






def standard_scaler(df_local, message = '', show_plots_local = False):
    scaler = preprocessing.StandardScaler()
    df_local_scaled = scaler.fit_transform(df_local[df_local.columns])
    df_local_scaled = pd.DataFrame(df_local_scaled, columns = df_local.columns)    
    
    if show_plots_local:
        plot_dataframe(df_local_scaled, message)
           
    return df_local_scaled


def min_max_scaler(df_local, message = '', show_plots_local = False):
    scaler = preprocessing.MinMaxScaler()
    df_local_scaled = scaler.fit_transform(df_local[df_local.columns])
    df_local_scaled = pd.DataFrame(df_local_scaled, columns = df_local.columns)    
    
    if show_plots_local:
        plot_dataframe(df_local_scaled, message)
           
    return df_local_scaled




























 

class plotter:

    def plot_cat_cat(x,y):            
        message = 'Plotter cat-cat' + x.name + 'vs target '+ y.name
        print(message)
        
        plt.figure()
        ax = sns.catplot(x= x, hue = y, kind = 'count', height=6)     
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
        plt.tight_layout()
        

        file_name = os.path.join(results_dir + '/' + message + '.jpg')
        plt.savefig(file_name, dpi = 1200)
        
        plt.show()
    
        
    def plot_cat_cont(x,y, kde = True):
        message = 'Plotter cat-cont  ' + x.name + 'vs target '+ y.name
        print(message)
        plt.figure()
        
        temp_df = pd.DataFrame()        
        temp_df[x.name] = x
        temp_df[y.name] = y
        
        sns.FacetGrid(temp_df, hue = y.name, height=6).map(sns.kdeplot, y.name , vertical = False).add_legend()   

#        sns.FacetGrid(x, hue = y, height=6).map(sns.kdeplot, x).add_legend() 

      
        plt.tight_layout()
        
        file_name = os.path.join(results_dir + '/' + message + '.jpg')
        plt.savefig(file_name, dpi = 1200)
        
        plt.show()
        
    
    
    def plot_cont_cont(x,y):
        message = 'Plotter cat-cont  ' + x.name + 'vs target '+ y.name
        print(message)
        plt.figure()
#        sns.FacetGrid(train, hue = "Survived", height=6).map(sns.kdeplot, column).add_legend()          
        ax = sns.scatterplot(x = x, y = y, hue = y)
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
        plt.tight_layout()
        
        file_name = os.path.join(results_dir + '/' + message + '.jpg')
        plt.savefig(file_name, dpi = 1200)
        
        plt.show()

    def plot_cont_cat(x,y):
        message = 'Plotter cat-cont  ' + x.name + 'vs target '+ y.name
        print(message)
        plt.figure()      
        
        #To Do: resolve this workaround or use plotly. Facetgrid.map doesnt work without this
        temp_df = pd.DataFrame()
        
        temp_df[x.name] = x
        temp_df[y.name] = y
        
#        sns.FacetGrid(x, hue = x, height=6).map(sns.kdeplot, y, vertical = False).add_legend()  
        sns.FacetGrid(temp_df, hue = x.name, height=6).map(sns.kdeplot, y.name , vertical = False).add_legend()   

        
        
#        sns.FacetGrid(data_df, hue = x, height=6).map(sns.kdeplot, y).add_legend()  
        
        plt.tight_layout() 
        
        file_name = os.path.join(results_dir + '/' + message + '.jpg')
        plt.savefig(file_name, dpi = 1200)
        
        plt.show()

#plotter.plot_cont_cont(x=train['MSSubClass'], y=y_train)
#
##train = train.drop(columns = target_column)
#plotter.plot_cont_cat(x=train['SaleType'], y=y_train)
#
#train[train.index.duplicated()]




#Target Analysis. Assuming univariate target
# The following is not working
#def target_analysis(y_series):    
#    
#    message = 'TARGET - kde plot - ' + y_series.name 
#    print(message)    
##    plt.figure()
##    
#    sns.kdeplot(y_series, shade = True, color = 'r')
#
##    plt.setp(ax.get_xticklabels(), rotation=45)
#    
##    plt.tight_layout()   
##
##    file_name = os.path.join(results_dir + '/' + message + '.jpg')
##    plt.savefig(file_name, dpi = 1200)
##    
#    plt.show()
#    
#    
#    
#target_analysis(y_train)



def seperate_cat_cont_columns(train):

    #to do: only categorical and unresolved seem to be used. delete extra entries. or have continuous entry
    categorical_columns = []
    continuous_columns = []
    unresolved_columns = []
    error_columns = []
    

    for idx, column in enumerate(train.columns):
        print()
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>',train.columns)
        
    
        if column == target_column:
            continue
        
        
        try:
            #This is needed because in regular passing of dataframe, it is passed as int64
            #but when passed by parallel processing library, it gets changed to Object
            #this step changes it back to int64
            train[column] = pd.to_numeric(train[column])
            is_numeric = True
            
        except: 
            is_numeric = False

    
        try:
            group_obj = train.groupby(column)
            group_count_in_column = group_obj.count().shape[0]
                       
            log_column_str = 'Log: >>>>> ' + '[' +  column + '] '

            
            if group_count_in_column <= max_groups_in_categorical_data:
                print('Log: [', column,'] is categorical. Unique groups in category: ', group_count_in_column)
                
                if is_numeric:
                    categorical_columns.append(column)

                    if column != target_column:
                        if show_plots:

                            if global_problem_type == 'categorical':
                                plotter.plot_cat_cat(x=train[column] , y = y_train)  
                                
                            elif global_problem_type == 'regression':
                                plotter.plot_cont_cat(x = train[column] , y  = y_train)  
                            else:
                                raise Exception('target is neither categorical nor regression')
                        
                else:
                    unresolved_columns.append(column)
                    print(log_column_str + 'is alphanumeric. Added for auto processing')
                    pass     
                
                    
            else:    
                if is_numeric and column != target_column:     

                    continuous_columns.append(column)               
                    
                    try:
                            if show_plots:

                                if global_problem_type == 'categorical':
                                    plotter.plot_cat_cont(x = train[column] , y  = y_train) 
                                    
                                elif global_problem_type == 'regression':
                                    plotter.plot_cont_cont(x = train[column] , y  = y_train) 
                                else:
                                    raise Exception('target is neither categorical nor regression')

    
                    except:
                            warn('Log: >>>>> Unknown error: Cannot make graph for column: ' + column)
                            error_columns.append(column)
    
                else:
                    print(log_column_str , 'Cannot make graph for continous column and object. Will be processed as text later')
                    unresolved_columns.append(column)
    
        except:
            print(log_column_str , 'Preprocessing failed. Proceeding to next column')
        

    return     categorical_columns, continuous_columns, unresolved_columns,  error_columns
        























































categorical_columns, continuous_columns, unresolved_columns,  error_columns = parallel.compute(train, function = seperate_cat_cont_columns)

#assert(len(categorical_columns) + len(continuous_columns) + len(unresolved_columns) == train.shape[1] )






train_categorical = train.loc[ :, categorical_columns]
train_continuous = train.loc[ :, continuous_columns]
train_unresolved = train.loc[ :, unresolved_columns]


        
#print('Log: ' , unresolved_columns, 'Could not be resolved as they were continous and of type object. These will be taken as string')       
plt.show()
print()
print()
















print('Cleaning text in unresolved columns')


# Clean text columns


def clean_unresolved_columns(df):
    auto_generated_data_df = pd.DataFrame()
    column_split_mapper_dict = {}

    for idx, column in enumerate(df.columns):
        print(column)
        if column == 'Electrical':
            print('Found')
        
        # first split with space, then remove special charecters
#        cleaned_and_split_df  = train[column_name_in_train_df].apply(clean_and_split_text)
        
        cleaned_and_split_df  = df[column].apply(clean_text, filters = '\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
                                              
        cleaned_and_split_df  = cleaned_and_split_df.apply(split_text, split_at = ' ')  
        
        cleaned_and_split_df = cleaned_and_split_df.apply(split_at_char_int_transistion)
        

            
        
        #convert column of lists into its own seperate column
        multi_column_df = cleaned_and_split_df.apply(pd.Series)
        
        #rename columnn names to include original name
        multi_column_df = multi_column_df.add_prefix(column + '_')           
        column_split_mapper_dict[column] = list(multi_column_df.columns)

    



        #pd.concat(auto_generated_categories, tags)
        auto_generated_data_df = auto_generated_data_df.join(multi_column_df, how = 'outer')


    return auto_generated_data_df, column_split_mapper_dict



auto_generated_data_df, column_split_mapper_dict = parallel.compute(train_unresolved, clean_unresolved_columns)







print()
print('Finding possible groups in cleaned data')


 
auto_generated_data_df.columns
#auto_generated_data_df_raw = auto_generated_data_df
auto_generated_data_df_categorical = pd.DataFrame(index=range(train.shape[0]))

#this is only to generate graphs. seaborn needs same dataframe for faced grids
#auto_generated_data_df[target_column] = train[target_column]

auto_generated_hash_df = pd.DataFrame()
auto_generated_data_df_continuous = pd.DataFrame()





#convert possible numbers to numeric type
def try_convert_to_numeric(df):
    for column in df.columns:
        #try to convert columns with all numbers to integers
        for row in df[column]:
    
            try:
                df.iloc[row,column] = pd.to_numeric(df.iloc[row,column])
                
                print('Converted auto generated column to int: ', column)
            except Exception:
                pass     
    return df


auto_generated_data_df = parallel.compute(auto_generated_data_df, try_convert_to_numeric)






new_expanded_df = pd.DataFrame(index=range(0,auto_generated_data_df.shape[0]))


def get_type(data):
    

    try:
        #while data is turned before to numeric, when it is passed by apply, it changes to float
        data = float(data)
    except Exception:
        pass     

    if type(data) == float:
        if math.isnan(data):
            return np.nan

    return str(type(data))
    
    

#### Tp Do : Do DP here
for column in column_split_mapper_dict.keys():
    data_types = set()

    all_sub_columns = column_split_mapper_dict[column]
    
    
    for sub_column in all_sub_columns:
        

        temp_df = pd.DataFrame(index=range(0,auto_generated_data_df[sub_column].shape[0]))

        
        print (column, '>',sub_column)

        
        temp_df['datatype'] = auto_generated_data_df[sub_column].apply(get_type)
        
        group_obj = temp_df.groupby('datatype') 
        group_dict = group_obj.groups

        
        ############# Work Here
        if group_obj.size().shape[0] > 1 :
            
            for key in group_dict.keys():
                
                if key!= 'NAN_DATA':
                    new_column_name = sub_column + '_' + str(key)
                    
                    
                    rows = group_dict[key]
                    
                    new_expanded_df.loc[rows, new_column_name] = auto_generated_data_df.loc[rows, sub_column]
                    
                    
                    #drop the column if its all nan
                    #new_expanded_df[new_column_name] = new_expanded_df[new_column_name].dropna()
                    
    #                if new_expanded_df[new_column_name] 
    
    #                
    #                temp_df.loc[rows, new_column_name] = auto_generated_data_df.loc[rows, sub_column]
                
                 
                print (group_dict.keys())
        
        
    print('longest entry: ', all_sub_columns[-1] )
    




group_dict_lookup_dict = {}

for column in auto_generated_data_df.columns:
        
    if column == target_column:
        continue
   
    found_new_feature = False
    
    group_obj = auto_generated_data_df.groupby(column)
    group_dict = group_obj.groups
    group_names = group_dict.keys()   
    
    total_groups = group_obj.size().shape[0]
    
    
    column_str = '[' + column +']'    
    if column == 'Name_1':
        print('Found' )

    group_dict_lookup_dict[column] = group_dict
    
    
    if total_groups <= max_groups_in_categorical_data:
        auto_generated_data_df_categorical[column] = np.nan
        
        for key in group_dict.keys():        
            if is_valid(key):
                members_in_group = group_dict[key].shape[0]
                rows_to_discard = group_dict[key]
                
                if members_in_group > min_members_in_a_group:   
                    
                    
                    # This if statement is to only print "Found groups in column:" once per column for which groups were found
                    if not found_new_feature:
                        print('Found groups in column: ', column_str, ' unique categories: ', total_groups)    
                    
                    found_new_feature = True
    
                    print('>>> Group Name:     ' , key, '    members of caregory: ', members_in_group )
                    auto_generated_data_df_categorical.loc[rows_to_discard, column] = key  
#                else:
#                    #mark all members with few members as discarded
#                    rows_to_discard = group_dict[key]
#                    auto_generated_data_df_categorical.loc[rows_to_discard, column] = np.nan   
                    
        #To Do: Figure out if this dropping is a good idea. Also add it to drop dataframe
        if auto_generated_data_df_categorical[column].count() == 0:
            drop_and_log_column(auto_generated_data_df_categorical, column, 'Zero Data')
#            auto_generated_data_df_categorical.drop(columns = [column])
            print('dropped empty column:', column)
        ### To Do: Plot graphs later        
        if found_new_feature:            
            if show_plots: 
                if global_problem_type == 'categorical':
                    plotter.plot_cat_cat( x = auto_generated_data_df[column] , y =  y_train )
                    
                elif global_problem_type == 'regression':  
                    plotter.plot_cont_cat( x = auto_generated_data_df[column] , y =  y_train )
                    
                else:
                    raise Exception('target is neither categorical nor regression')
                    
#                plt.figure()
#                ax = sns.countplot(auto_generated_data_df_categorical[column], hue = train[target_column])
#                ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
#                plt.tight_layout()
#                plt.show()
            print()
            print()
            print()

    else:

        if auto_generated_data_df[column].count() >= auto_min_entries_in_continous_column: 
            
            potential_continuous_column =  auto_generated_data_df[column].copy(deep = True)
                 
            try:                
                auto_generated_data_df_continuous[column] = pd.to_numeric(potential_continuous_column)
                print(column_str, 'Found new auto generated continuous numeric column')
    
    
            except Exception:
                #reversing the add that was done before the            
#                auto_generated_data_df_continuous = auto_generated_data_df_continuous.drop(columns = [column])

#                print(column_str, 'Dropping auto-generated continuous column as it is string. Adding to auto_generated_dropped & auto_generated_hash_df')
#                auto_generated_data_df_dropped = auto_generated_data_df[column].copy(deep = True)
                
                  
                print('Applying hash trick to: ', column)
                print('Type: ', type(auto_generated_data_df[column][1]))
                auto_generated_hash_df[column] = auto_generated_data_df[column].copy(deep = True)
                auto_generated_hash_df[column] = auto_generated_hash_df[column].apply(str)
                
                #by this point, it is a string and not an int. so we use the first letter as hash function
                auto_generated_hash_df[column] = auto_generated_hash_df[column].apply(hash_it, strategy = hashing_strategy)   
                
                
                
                
                pass

        else:
            print(column_str, 'Too little data for continuous')
            drop_and_log_column(auto_generated_data_df[column])
            
            
#To Do: These need to be added to auto_generated_data_df_dropped
print('Dropping auto generated columns with no features')
auto_generated_data_df_categorical = auto_generated_data_df_categorical.dropna(axis='columns', how = 'all')            
            
#            sns.FacetGrid(train, hue = "Survived", height=6).map(sns.distplot, 'temp', kde = True).add_legend()             
#            sns.FacetGrid(train, hue = "Survived", height=6).map(sns.distplot, 'temp', kde = False).add_legend()  


#column_split_mapper_dict
#group_dict_lookup_dict
#
#for main_column in column_split_mapper_dict.keys():
#    for sub_column in column_split_mapper_dict[main_column]:
#        print(main_column,  '>', sub_column, '  ', group_dict_lookup_dict[sub_column].keys() )
#        
#        
#    print()
#
#            


print('Showing plots for auto-generated Continuous Data')
#this is only to generate graphs. seaborn needs same dataframe for faced grids
auto_generated_data_df_continuous[target_column] = train[target_column]
auto_generated_data_df_continuous[target_column] = pd.to_numeric(auto_generated_data_df_continuous[target_column]).copy(deep = True)

for column in auto_generated_data_df_continuous.columns:
    if column == target_column:
        continue
    print(column)
#    ax = sns.countplot(auto_generated_data_df_continuous[column], hue = train[target_column])
    
    try:
        auto_generated_data_df_continuous[column] = pd.to_numeric(auto_generated_data_df_continuous[column])
        
        if show_plots:
            if global_problem_type == 'categorical':
#                plotter.plot_cat_cat(auto_generated_data_df , x = column , y = target_column, log_column_str = '', log_target_str = '')
                plotter.plot_cat_cont(x = auto_generated_data_df[column] , y = y_train, kde = True)
                plotter.plot_cat_cont(x = auto_generated_data_df[column] , y = y_train, kde = True)
                
            elif global_problem_type == 'regression':  
                plotter.plot_cont_cont(auto_generated_data_df[column] , y = y_train)
                
            else:
                raise Exception('target is neither categorical nor regression')
                    
                
#            sns.FacetGrid(auto_generated_data_df_continuous, hue = target_column, height=6).map(sns.distplot, column, kde = True).add_legend()             
#            sns.FacetGrid(auto_generated_data_df_continuous, hue = target_column, height=6).map(sns.distplot, column, kde = False).add_legend()     
#           
#            plt.tight_layout()
#            plt.show()        
    
    except Exception:
        print('There is non numeric data in auto_generated_data_df_continuous. This should not have happened')
           
auto_generated_data_df_continuous = auto_generated_data_df_continuous.drop(columns = [target_column])
        


print('Showing plots for auto-generated Hashed Data')
for column in auto_generated_hash_df.columns:
    if column == target_column:
        continue
    
    try:
        if show_plots:
            ax = sns.countplot(auto_generated_hash_df[column], hue = train[target_column])
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
            plt.tight_layout()
            plt.show() 
            
            if global_problem_type == 'categorical':
                plotter.plot_cat_cat(x = auto_generated_data_df[column], y = y_train)
                
            elif global_problem_type == 'regression':  
                plotter.plot_cont_cat(auto_generated_data_df[column] , y = y_train)
                
            else:
                raise Exception('target is neither categorical nor regression')
                
            
    except Exception:
        pass  
    


def one_hot(df, exclude_from_ohe = None):
    cat_columns = df.columns
    
    
    
    df_excluded_from_one_hot = pd.DataFrame()
    
    df_after_one_hot = pd.DataFrame()

    if exclude_from_ohe == None:
        df_after_one_hot = pd.get_dummies(df, columns = df.columns)
    
    else:         
        columns_to_one_hot = []
        
        for column_name in cat_columns:
            if column_name in exclude_from_ohe:
                df_excluded_from_one_hot[column_name] = df[column_name].copy(deep = True)
                warn(column_name + ': excluded from one hot in cat_preprocess')          
            else:
                columns_to_one_hot.append(column_name)
    
        #perform one hot encoding    
        if len(columns_to_one_hot) != 0 :
            df_after_one_hot= pd.get_dummies(df, columns= columns_to_one_hot )
           
          
    return df_after_one_hot



def show_heatmap(X_train, message = '', x_label = '', y_label = '', show_absolute_values = True):
    
    #Show corelated features . Dont use np.corrcoef as it is not nan tolerant
    print('Corelation (positive & negative) using numpy')
    corr = np.corrcoef(X_train.values, rowvar=False)
    corr = np.absolute(corr)
    
    sns.heatmap(corr)  
    plt.show()   
    
    
    corr_df = X_train.corr()
    
    
    if show_absolute_values:
        #convert 
        corr_df = corr_df.select_dtypes(include=[np.number]).abs()        
        message = message + '_absolute_correlation'
    
#    corr = np.corrcoef(X_train)
#    corr_df = pd.DataFrame(corr)
    
#    corr_df = pd.DataFrame(corr, columns = X_train.columns)

#    labels = list(corr_df.columns.values)
#    trace = go.Heatmap( z=corr_df, x = labels , y = labels , colorscale='Viridis')

    
    trace = go.Heatmap( z = corr_df.values.tolist(), x = corr_df.columns , y = corr_df.columns , colorscale='Viridis')
#    trace = go.Heatmap( z = corr_df.values.tolist() , colorscale='Viridis')
    
    
    traces = [trace]
    
    layout = go.Layout(         
        title=dict(
                    text = message,
                    
                    font=dict(
                                family='Courier New, monospace',
                                size=48,
                                color='#7f7f7f'
                            )
                    ),   
                    
        xaxis=dict(
                    title = x_label,
                    
                    titlefont=dict(
                                    family='Courier New, monospace',
                                    size=18,
                                    color='#7f7f7f',
                                    
                                ),
                    tickmode = 'linear'
                    ),
                    
        yaxis=dict(
                    title= y_label,
                    titlefont=dict(
                                    family='Courier New, monospace',
                                    size=18,
                                    color='#7f7f7f',
                                ),
                    tickmode = 'linear',
                    autorange='reversed'
                    )
    )
        
    
    fig = go.Figure(data=traces, layout=layout)
    
    filename = os.path.join(results_dir, message +  '.html')   
    plotly.offline.plot(fig, show_link = True, filename = filename)



                    
def profiler_analysis(df1, df2 = None, message = ''):
    if df2 is not None:
        combined_df = df2.join(df1,how = 'outer')
    else:
        combined_df = df1

    profile_combined = pandas_profiling.ProfileReport(combined_df, title = message + ' Profiler Analysis ', pool_size = 0)
    
    output_html_file = os.path.join (results_dir , "Profile_Report_Before_Imputation.html")                                       
    profile_combined.to_file(output_file= output_html_file) 
    webbrowser.open('file://' + output_html_file)


    profiler_results = profile_combined.get_description()
    

    return profiler_results





def pre_processing(categorical_df, continuous_df, imputer, enable_ohe, exclude_column_from_ohe):   

    #label encode categorical
    
    categorical_df_label_encoded =  label_encoder(categorical_df, strategy = 'keep_missing_as_nan')
    #here outer is used. so continuous
    cat_columns = categorical_df_label_encoded.columns
    cont_columns = continuous_df.columns
    
    joint_df = categorical_df_label_encoded.join( continuous_df, how = 'outer')      
    
    
    if show_plots:
        show_heatmap(joint_df, message = 'heatmap_before_imputation')

    #get indexes of categorical columns. this is used by miss_forest to identify categorical features
    cat_column_indexes_in_joint_df = []    
    for column in cat_columns:
        cat_column_indexes_in_joint_df.append(joint_df.columns.get_loc(column))
    
    
    cat_column_indexes_in_joint_df = np.asarray(cat_column_indexes_in_joint_df)
    
    
    if imputer == 'random_forest':
        print('Using random forest as imputer')
        imp = MissForest(max_iter = 2, n_estimators = 10, n_jobs = 24, verbose = 0)
        imputed_df = imp.fit_transform(joint_df, cat_vars = cat_column_indexes_in_joint_df )
        #imputation leads to loss of column information. this step restores column names and gives back dataframe
        imputed_df = pd.DataFrame(imputed_df)
        imputed_df.columns = joint_df.columns
        
    elif imputer == 'knn':
        col_max_missing = 1        
        if col_max_missing > min_data_in_column_percent:
            warn('manual override: There were too many missing columns for knn imputation. col_max_missing = '+ str(col_max_missing * 100) + ' percent' + '   min_data_in_column_percent:' + str(min_data_in_column_percent))
        
        imp = KNNImputer(col_max_missing=col_max_missing)
        imputed_df = imp.fit_transform(joint_df)    
        imputed_df = pd.DataFrame(imputed_df)
        imputed_df.columns = joint_df.columns
    
    elif imputer == None:
        #no imputation
        imputed_df = joint_df
        
    else:
        warn('incorrect imputer')
        raise Exception('Incorrect imputer specified. Use None for no imputation')
    
    
    
        
    # remove features with zero varience
    selector = feature_selection.VarianceThreshold()
    selector.fit(imputed_df)
    
    supports = selector.get_support()
    indices_with_zero_variance = np.where(supports == False)
    indices_with_zero_variance = list(indices_with_zero_variance[0])    
    columns_to_drop = imputed_df.columns[indices_with_zero_variance]
    
    
    cat_columns_list_after_feature_removal = list(cat_columns)
    cont_columns_list_after_feature_removal = list(cont_columns)
    
    
    for column in columns_to_drop:
        drop_and_log_column(imputed_df, column, '_Zero variance after imputation')
        warn(str(column) + ' Dropped column due to zero varience after imputation. Maybe its better to make missing values as unknown')
        
        if column in cat_columns_list_after_feature_removal:
            cat_columns_list_after_feature_removal.remove(column)
        elif column in cont_columns_list_after_feature_removal:
            cont_columns_list_after_feature_removal.remove(column)
    
    
    if show_plots:
        show_heatmap(imputed_df, message = 'heatmap_after_imputation')
        

    
    
    if enable_ohe:
        #the cat and cont are seperated and then joint so that the order is preserved after one hot
        imputed_cat_df = imputed_df[list(cat_columns_list_after_feature_removal)]
        imputed_cont_df = imputed_df[list(cont_columns_list_after_feature_removal)]
        
        
        #one hot categorical columns    
        imputed_cat_df = one_hot(imputed_cat_df, exclude_column_from_ohe)
        
        imputed_df = imputed_cat_df.join(imputed_cont_df,how = 'outer')

    
    
    return imputed_df
    



combined_categorical_df = pd.DataFrame()
combined_categorical_df = combined_categorical_df.join(train_categorical,how = 'outer')
combined_categorical_df = combined_categorical_df.join(auto_generated_data_df_categorical,how = 'outer')
combined_categorical_df = combined_categorical_df.join( auto_generated_hash_df, how = 'outer')
#combined_categorical_df = combined_categorical_df.astype('category')



combined_continuous_df = pd.DataFrame()
combined_continuous_df = combined_continuous_df.join(train_continuous,how = 'outer')
combined_continuous_df = combined_continuous_df.join(auto_generated_data_df_continuous,how = 'outer')



def box_plot(df_local, message = ''):
    df_local.iplot(kind='box', boxpoints='outliers', title = message + ' - Box Plot')
    
#box_plot(combined_categorical_df)







def extract_data_from_profiler_messages(profiler_results, value_to_extract, extract_from = 'variables', message = '', y_title = '', is_percent = False ):
    
    extracted_data_dict = {}
    
    for res in profiler_results[extract_from]:
#        feature_name = message.model.messages.Message.column_name
#        column_name = message.column_name
        feature_name = res.values['varname']
        
        print('>>>', feature_name)
        
        if feature_name == 'Electrical_0':
            print('found')
        
        try:
            value = res.values[value_to_extract]
        except:
            #if the column is string, the P_zeros is not populated. 
            #hence zero_percentage is set to 0
            value = 0
            print(feature_name)

        if value != 0:
            extracted_data_dict[feature_name] = value

#        print(column_name)
#        analysis_df.loc[column_name, 'missing'] = missing_percentage
#        analysis_df.loc[column_name, 'zero_percentage'] = zero_percentage
#        
#        
    
    extracted_data_dict_df =  pd.DataFrame.from_dict(extracted_data_dict, orient = 'index')
    extracted_data_dict_df.columns = [value_to_extract]
    
    if is_percent:
        extracted_data_dict_df = extracted_data_dict_df*100
        y_title = y_title + 'Percentage - Range 0 to 100' 
    
    extracted_data_dict_df = extracted_data_dict_df.sort_values(by = value_to_extract, ascending = False)    
    extracted_data_dict_df.iplot(kind='bar', yTitle=y_title, title= message, filename= results_dir + message)
    
    
    
    
    return extracted_data_dict_df
    



def retrive_from_profiler_results(profiler_results_df, value_to_retrive, title_for_graph, is_percent = False, y_title = ''):
        res = profiler_results_df[value_to_retrive]
        res = res.where(res != 0)
        res.dropna(inplace = True) 
        res.sort_values(ascending = False, inplace = True) 
        
        
        if is_percent:            
            res = res * 100 
            y_title = y_title + 'Percentage - Range [0 - 100]'
            
        filename = os.path.join(results_dir,  title_for_graph)
        
        res.iplot(kind='bar', yTitle=y_title, title= title_for_graph, filename=filename )
        
        return res
    
    
def analyse(combined_categorical_df, combined_continuous_df, message = ''):
    
#    #
#    message = 'scaled_standard'
#    combined_continuous_df_scaled = standard_scaler(combined_continuous_df, message, show_plots_local = False)
#    

        
#    combined_categorical_df.iplot(kind='box', boxpoints='outliers', title = 'Box Plot - Before Preprocessing - combined_categorical_df')
#    combined_continuous_df.iplot(kind='box', boxpoints='outliers', title = 'Box Plot - Before Preprocessing - combined_continuous_df')
    
    box_plot(combined_categorical_df, message = 'Box Plot - Before Preprocessing - combined_categorical_df')
    box_plot(combined_continuous_df, message = 'Box Plot - Before Preprocessing - combined_continuous_df')
    
    profiler_results = profiler_analysis(combined_categorical_df, combined_continuous_df, message = 'Profiler Before Imputation')
    
    

#    value_to_extract = 'p_missing'
#    extract_data_from_profiler_messages(profiler_results, value_to_extract, message = message + ' Features with Missing Data', is_percent = True)
#
#    value_to_extract = 'p_zeros'
#    extract_data_from_profiler_messages(profiler_results, value_to_extract, message = message + ' Features with Zeros', is_percent = True)

    profiler_results_df =  pd.DataFrame.from_dict(profiler_results['variables'], orient = 'index')
    
    
    results_df = pd.DataFrame()

    res = retrive_from_profiler_results(profiler_results_df, value_to_retrive = 'p_missing', title_for_graph = message + ' - Missing Data - Percentage', is_percent = True)
    results_df = results_df.join(res,how = 'outer')    
    
    res = retrive_from_profiler_results(profiler_results_df, value_to_retrive = 'p_zeros', title_for_graph = message + ' - Features with Zeros - Percentage', is_percent = True)
    results_df = results_df.join(res,how = 'outer')    

    res = retrive_from_profiler_results(profiler_results_df, value_to_retrive = 'p_unique', title_for_graph = message + ' - Unique Data - Percentage', is_percent = True)
    results_df = results_df.join(res,how = 'outer')    
 
    res = retrive_from_profiler_results(profiler_results_df, value_to_retrive = 'distinct_count', title_for_graph = message + ' - Unique Data Counts', is_percent = False)
    results_df = results_df.join(res,how = 'outer')    
 
    #check for features that have both missing data and zeros
    features_with_zeros_and_missing_data = list(set(results_df.p_missing.dropna().index) & set(results_df.p_zeros.dropna().index))

    if len(features_with_zeros_and_missing_data) != 0:
        warn('Feature has both zeros and nan - '+ str(features_with_zeros_and_missing_data) )
        
        
    
        
    
    
    return profiler_results, results_df, features_with_zeros_and_missing_data



    



#counts = 0 
#def missing_data_analysis(df_local):
##    for column in df_local.columns:
#        column = 'MSZoning'
#        group_obj = df_local.groupby(column)
#        group_dict = group_obj.groups
#        
#        max_count = 0
#        max_key = None
#        
#        for key in group_dict.keys():            
#            count = group_dict[key].shape[0]
#            
#            print('>>>' , count)
#            
#            if count > max_count:
#                max_count = count
#                max_key = key
#                
#                
#            
#        return counts
#
#missing_data_analysis(train)




#profile_combined_categorical = pandas_profiling.ProfileReport(combined_categorical_df, title = 'Categorical Variables', pool_size = 0)
#profile_combined_categorical.to_file(output_file= os.path.join (results_dir , "Categorical_Variables_Profile_Report.html"))
#rejected_variables = profile_combined_categorical.get_rejected_variables(threshold=0.9)
#
#
#profile_combined_continuous_df = pandas_profiling.ProfileReport(combined_continuous_df, title = 'Continuous Variables', pool_size = 0)
#profile_combined_continuous_df.to_file(output_file= os.path.join (results_dir , "Combined_Continuous_df_Profile_Report.html"))
#rejected_variables = profile.get_rejected_variables(threshold=0.9)
#




#combined_continuous_df.profile_report(correlations={"cramers": False})












































































































#Analysis
    
#Target Analysis























profiler_results, profiler_results_df, features_with_zeros_and_missing_data = analyse(combined_categorical_df, combined_continuous_df, message = 'Before Pre Processing')
#To Do: find why the rejected variable list is empty and create a dataset without those variables





show_plots = True
X_train = pre_processing(combined_categorical_df, combined_continuous_df, imputer = 'random_forest', enable_ohe = False, exclude_column_from_ohe = exclude_from_ohe)


profiler_analysis(X_train, message = 'Profiler Before Imputation')

plt.figure()
X_train.boxplot()
plt.show()









X_train_dict['original'] = X_train.copy(deep = True)




#
#
#tree



####################################################################################################3

use_dtree = False
if use_dtree:
    dtree = tree.DecisionTreeClassifier()
    dtree.fit(X_train, y_train)
    
    dtree_pre_analysis = analyze_tree(dtree, X_train, y_train, max_depth_search_list_length = 10)

    dtree = tree.DecisionTreeClassifier(presort = True)
    
    dt_param_grid = {
            'criterion': ['gini','entropy'] ,
            'max_depth': dtree_pre_analysis['max_depth_search_list']  ,
            'min_samples_split': list(range(2,50,1)),
            'min_samples_leaf': list(range(1,50,1)),
            }
    
    dt_grid_estimator = model_selection.GridSearchCV(dtree, dt_param_grid, scoring = 'accuracy', n_jobs = -1, refit = True, verbose = 1, return_train_score = True)
    
    dt_grid_estimator.fit(X_train, y_train)
    
    results_dt_grid_estimator = dt_grid_estimator.cv_results_
    
    dt_grid_estimator.best_score_
    dt_grid_estimator.best_params_
    
    
    dtree_post_analysis = analyze_tree(dt_grid_estimator.best_estimator_, X_train, y_train)
    

####################################################################################################3




#Scale data for KNN. Here scaling should only be done for continuous. 
#Doing it on categorical isnt harmful but wastes CPU compute
#it is only done for simplicity as dataset is very small

#To DO: Find which columns to drop. Performance improves by dropping columns


#scaler = preprocessing.StandardScaler()
#X_train = scaler.fit_transform(X_train[X_train.columns])
#
#knn = neighbors.KNeighborsClassifier()
#
#    
#n_neighbours = get_equally_spaced_numbers_in_range(1, (X_train.shape[0]) /10, 100) 
#    
## 'algorithm': ['ball_tree', 'kd_tree', 'brute'] ,
#
#knn_param_grid = {
#                    'n_neighbors' : n_neighbours , 
#                    'weights' : ['uniform', 'distance'] ,
#                }
#knn_grid_estimator = model_selection.GridSearchCV(knn, knn_param_grid,  return_train_score = True, scoring = 'accuracy', n_jobs = -1)
#
#knn_grid_estimator.fit(X_train, y_train)
#
#results = knn_grid_estimator.cv_results_
#
#print('Best Knn Score: ', knn_grid_estimator.best_score_)
#
#print('mean_train_score: ', results.get('mean_train_score').mean())
#print('mean_test_score: ', results.get('mean_test_score').mean())
#
#
#knn_grid_estimator_results = knn_grid_estimator.cv_results_['params']
#
#knn_grid_estimator.best_params_
#
#
#
#X_train_dropped = X_train
#


###############################################################################################

#BAG Ensamble
use_bag_tree = False
if use_bag_tree:
    dtree = tree.DecisionTreeClassifier()
    
    bag_ensemble = ensemble.BaggingClassifier(base_estimator = dtree)
    
    bag_grid = {
                    'n_estimators' : get_equally_spaced_numbers_in_range(800,2000),
                    'base_estimator__criterion': ['gini','entropy'] ,
                    'base_estimator__max_depth' : dtree_pre_analysis['max_depth_search_list'],
    #                'base_estimator__min_samples_split' : list(range(2,50,1)) ,
    #                'base_estimator__min_samples_leaf' :  list(range(1,50,1))                                               
            }
    
    bag__tree_grid_estimator = model_selection.GridSearchCV(bag_ensemble, bag_grid, scoring = 'accuracy', n_jobs = -1, refit = True, verbose = 1, return_train_score = True, cv =10)
    
    bag__tree_grid_estimator.fit(X_train, y_train)
    
    print(bag__tree_grid_estimator.best_score_)
    print(bag__tree_grid_estimator.best_params_)
    final_estimator = bag__tree_grid_estimator.best_estimator_
    final_estimator.score(X_train, y_train)



### BAG Ensamble with KNN
use_bag_knn = False

if use_bag_knn:    
    knn_estimator = neighbors.KNeighborsClassifier()
    
    knn_bag_ensemble = ensemble.BaggingClassifier(base_estimator = knn_estimator)
    
    knn_bag_ensemble_param_grid = {
                                        'n_estimators' : get_equally_spaced_numbers_in_range(1,100) ,
                                        'base_estimator__n_neighbors': get_equally_spaced_numbers_in_range(1, (X_train.shape[0]) /10, 100)  
                                }
    
    bag_knn_grid_estimator = model_selection.GridSearchCV(estimator = knn_bag_ensemble, param_grid = knn_bag_ensemble_param_grid, scoring = 'accuracy', n_jobs = -1, refit = True, verbose = 1, return_train_score = True, cv =10)
    
    bag_knn_grid_estimator.fit(X_train, y_train)

    print('Best SCore: ', bag_knn_grid_estimator.best_score_)
    print('Best Params: ', bag_knn_grid_estimator.best_params_)



### Random Forest
use_random_forest = False
if use_random_forest:
    rf_estimator = ensemble.RandomForestClassifier()
    
    rf_grid = {
                'n_estimators' : get_equally_spaced_numbers_in_range(1,10000) ,
                'max_depth' : dtree_pre_analysis['max_depth_search_list'],
                'max_features': ['sqrt', 'log2'],
                
#                'min_samples_split' : list(range(2,50,1)) ,
#                'min_samples_leaf' :  list(range(1,50,1))   
            }
    
    rf_grid_estimator = model_selection.GridSearchCV(rf_estimator, rf_grid, scoring = 'accuracy', n_jobs = -1, refit = True, verbose = 1, return_train_score = True, cv =10)
    
    rf_grid_estimator.fit(X_train, y_train)
    
    print('Best SCore: ', rf_grid_estimator.best_score_)
    print('Best Params: ', rf_grid_estimator.best_params_)
    #print('Best SCore: ', rf_grid_estimator.best_estimator_.feature_importances_)
    
    
    results_rf_grid_estimator = rf_grid_estimator.cv_results_





use_extra_trees = False
if use_extra_trees:
    et_estimator = ensemble.ExtraTreesClassifier()
    
    et_grid = {
                'n_estimators' : get_equally_spaced_numbers_in_range(1,10000) ,
                'max_depth' : dtree_pre_analysis['max_depth_search_list'],
                'max_features': ['sqrt', 'log2']
            }
    
    et_grid_estimator = model_selection.GridSearchCV(et_estimator, et_grid, scoring = 'accuracy', n_jobs = -1, refit = True, verbose = 1, return_train_score = True, cv =10)
    
    et_grid_estimator.fit(X_train, y_train)
    
    print('Best SCore: ', et_grid_estimator.best_score_)
    print('Best Params: ', et_grid_estimator.best_params_)
    #print('Best SCore: ', rf_grid_estimator.best_estimator_.feature_importances_)
    
    
    results_et_grid_estimator = et_grid_estimator.cv_results_





#####################################################################
# Boosting
  
# Adaboost with decision tree
    
def plot_estimator_results_3d(grid_estimator, plot_type = 'mesh'):
    cv_results_ = grid_estimator.cv_results_
    #params_df = cv_results_['params']
    params_df = pd.DataFrame(cv_results_['params'])
    
    all_param_names = params_df.columns
    
    all_param_names_excluding_base_estimator = []
    
    for name in all_param_names:
        # if '__' is not found, then append list
        if name.find('__') == -1:
            all_param_names_excluding_base_estimator.append(name)
    
    #remove params for base estimators
    
    
    if len(all_param_names_excluding_base_estimator) == 2:     
        x = params_df[all_param_names_excluding_base_estimator[0]]
        y = params_df[all_param_names_excluding_base_estimator[1]]
        
        train_scores_df = pd.DataFrame(cv_results_['mean_train_score'])
        train_scores = train_scores_df.iloc[:,0]
        
        test_scores_df = pd.DataFrame(cv_results_['mean_test_score'])
        test_scores = test_scores_df.iloc[:,0]
        
        if plot_type == 'mesh':           
            
            trace_train = go.Mesh3d(x=x,y=y,z=train_scores,
                       alphahull=3,
                       opacity=.5,
                       colorscale="Reds",
                       intensity=train_scores,                        
                       )
            
            trace_test = go.Mesh3d(x=x,y=y,z=test_scores,
                       alphahull=3,
                       opacity=.5,
                       colorscale="Greens",
                       intensity=test_scores,                        
                       )
            
            traces = [trace_train, trace_test]
        
        elif plot_type == 'scatter':
            trace_test = go.Scatter3d(
                x=x,
                y=y,
                z=test_scores,
                mode='markers',
                marker=dict(
                    size=12,
                    color=test_scores,                # set color to an array/list of desired values
                    colorscale='Viridis',   # choose a colorscale
                    opacity=0.5
                )
            )
                
            traces = [trace_test]
                   
        plotly.offline.plot(traces)
        
        
    else:
        print('Cannot plot 3d plot. Number of params are: ', len(all_param_names_excluding_base_estimator))

    return traces




def rmse(y_orig, y_pred):
    return math.sqrt(metrics.mean_squared_error(y_orig,y_pred) )

def log_rmse(y_orig, y_pred):
    return math.sqrt(metrics.mean_squared_log_error(y_orig,y_pred) )



def ada_boost(X_train, y_train, message = ''):
        if global_problem_type == 'classification':            
            base_classifier = tree.DecisionTreeClassifier()            
            ada_boost_dt = ensemble.AdaBoostClassifier(base_classifier)
            scoring = 'accuracy'                      
            
        else:
            base_classifier = tree.DecisionTreeRegressor()  
            ada_boost_dt = ensemble.AdaBoostRegressor(base_classifier)
            scoring = global_scoring
            
            

            

        
        # note: it is very important to specify max depth = 1 so only stumps are built. 
        # if not then whole tree is built causing overfitting. 
        
        ada_boost_grid = {
                            'base_estimator__max_depth' : [1], 
                            'n_estimators': get_equally_spaced_numbers_in_range(1,1000,10) ,
                            'learning_rate': get_equally_spaced_non_zero_floats_in_range(.001,.5,10) ,
                }
    
    #    ada_boost_grid = {
    #                        'base_estimator__max_depth' : [1], 
    #                        'n_estimators':[31],
    #                        'learning_rate': [.35] ,
    #            }
        
        ada_boost_grid_estimator = model_selection.GridSearchCV(ada_boost_dt, ada_boost_grid, scoring = scoring, n_jobs = -1, refit = True, verbose = 1, return_train_score = True, cv =10)
        
        
        ada_boost_grid_estimator.fit(X_train, y_train)
        
        print('Best SCore: ', ada_boost_grid_estimator.best_score_)
        print('Best Params: ', ada_boost_grid_estimator.best_params_)
        
        traces = plot_estimator_results_3d(ada_boost_grid_estimator)
        
        results_ada_boost_grid_estimator = ada_boost_grid_estimator.cv_results_

        name = 'ada_boost' + message        
        est = estimator()
        est.name = name
        est.estimator = ada_boost_grid_estimator
        est.best_estimator = ada_boost_grid_estimator.best_estimator_
        est.traces = traces
        est.best_score = ada_boost_grid_estimator.best_score_
        est.best_params = ada_boost_grid_estimator.best_params_  
       
        
        estimators_dict[name] = est
        
        return results_ada_boost_grid_estimator




#####
#gradient boost
#stacking
#voting


#####################
#PCA
    

def plot3d(x,y,z, color='rgb(204, 204, 204)', message = 'Title', x_label = '', y_label = ''):
    if type(color) == pd.DataFrame:
        color = color[0].values
        
    trace = go.Scatter3d(
        x = x,
        y = y,
        z = z,
        mode='markers',
        marker=dict(
            size=12,
                    # set color to an array/list of desired values
            color=color,
            colorscale='Viridis',   # choose a colorscale
            opacity=0.8
        )
    )
        
    traces = [trace]
    
    
    layout = go.Layout(         
        title=dict(
                    text = message,
                    
                    font=dict(
                                family='Courier New, monospace',
                                size=36,
                                color='#7f7f7f'
                            )
                    ),   
                    
        xaxis=dict(
                    title = x_label,
                    
                    titlefont=dict(
                                    family='Courier New, monospace',
                                    size=18,
                                    color='#7f7f7f'
                                )
                    ),
                    
        yaxis=dict(
                    title= y_label,
                    titlefont=dict(
                                    family='Courier New, monospace',
                                    size=18,
                                    color='#7f7f7f'
                                )
                    )
    )
        
    
    fig = go.Figure(data=traces, layout=layout)   
    filename = os.path.join(results_dir, message +  '.html')   
    plotly.offline.plot(fig, show_link = True, filename = filename)
    
    return traces


def plot2d(x,y, threshold = '', color = 'rgb(255, 215, 0)', message = 'Title', x_label = '', y_label = ''):
    if threshold != '':
        threshold = threshold + 1
        
        if threshold > len(x):
            threshold  = len(x)
        color = list(np.full(threshold, 1)) + list(np.full(y.shape[0] - threshold, 0))
        
        
    trace1 = go.Scatter(
        x = x,    
        y = y,
        mode='markers',
        marker=dict(
            size=16,
            color = color, #set color equal to a variable
            colorscale='Viridis',
            showscale=True
        )
    )
    traces = [trace1]     



    layout = go.Layout(         
        title=dict(
                    text = message,
                    
                    font=dict(
                                family='Courier New, monospace',
                                size=48,
                                color='#7f7f7f'
                            )
                    ),   
                    
        xaxis=dict(
                    title = x_label,
                    
                    titlefont=dict(
                                    family='Courier New, monospace',
                                    size=18,
                                    color='#7f7f7f'
                                )
                    ),
                    
        yaxis=dict(
                    title= y_label,
                    titlefont=dict(
                                    family='Courier New, monospace',
                                    size=18,
                                    color='#7f7f7f'
                                )
                    ),                   
                  
    )
                    
                    
    fig = go.Figure(data=traces, layout=layout)
    filename = os.path.join(results_dir, message +  '.html')   
    plotly.offline.plot(fig, show_link = True, filename = filename)
    
    return traces


def plot_series(y, threshold = '', message = ''):
    x_labels = list(range(0, y.shape[0]))
    plot2d(x_labels, y , threshold = threshold, message = message)



def iso_map(df, target , n_components = 3, show_graphs = True, message = ''):
    embedding = Isomap(n_components = n_components)
    
    reduced_dimention = embedding.fit_transform(df)
    reduced_dimention_df = pd.DataFrame(reduced_dimention)
    
    if show_graphs == True:        
        if n_components == 3:
            x=reduced_dimention_df[0]
            y=reduced_dimention_df[1]
            z=reduced_dimention_df[2]
            plot3d(x,y, z, color = target, message = 'Isomap on scaled')
            
        elif n_components == 2:
            x=reduced_dimention_df[0]
            y=reduced_dimention_df[1]
            plot2d(x,y, color = target, message = 'Isomap on scaled')
        
    return reduced_dimention_df



def reduce_dimentions(df, target , algorithm , pca_cumulative_ratio_threshold = None , n_components = None, perplexity = 30, show_graphs = True, learning_rate = 10, message = ''):
    if type(df) == pd.DataFrame:
        df = df.values
    if type(target) == pd.DataFrame:
        target = target[0].values
        
    if algorithm == 'pca':              
        if pca_cumulative_ratio_threshold != None and n_components != None :        
            warn('reduce_dimentions: cannot have both pca_cumulative_ratio_threshold and n_components. Prioritising pca_cumulative_ratio_threshold and setting n_components to full dimention of data')
            n_components = df.shape[1]
        elif n_components != None and pca_cumulative_ratio_threshold == None:
            if n_components > df.shape[1]:
                warn('PCA: n_components > feature columns in data. Setting n_components = features in data')
                n_components = df.shape[1]
            pca_cumulative_ratio_threshold = .999999
        elif pca_cumulative_ratio_threshold != None and n_components == None :
            n_components = df.shape[1]
        else:
            n_components = df.shape[1]
            pca_cumulative_ratio_threshold = .999999
        
        assert (pca_cumulative_ratio_threshold >= 0 and pca_cumulative_ratio_threshold <= 1)
        
        lpca = decomposition.PCA(n_components = n_components)
#        lpca.fit(df)       
        
        reduced_dimention_np_arr = lpca.fit_transform(df)        
        reduced_dimention_df = pd.DataFrame(reduced_dimention_np_arr)        
        lpca.explained_variance_ratio_        
        plot_series(lpca.explained_variance_ratio_, message = 'PCA: explained_variance_ratio_')
        cumulative_variance = np.cumsum(lpca.explained_variance_ratio_)        
        index_of_pca_first_redundant_pca_axis = np.searchsorted(cumulative_variance, pca_cumulative_ratio_threshold, side = 'left') 

        print('>>>>>>>' , index_of_pca_first_redundant_pca_axis)
        
        if max(cumulative_variance) >= pca_cumulative_ratio_threshold:
            print('Cumulative Varience Threshold of ', pca_cumulative_ratio_threshold, '  achieved by PC axis at index ', index_of_pca_first_redundant_pca_axis, ' where last PC index is ', df.shape[1]- 1  )
        else:
            warn('Cannot capture all varience in ' +  str(n_components) + ' PCA components. Try increasing PCA components.' +
                 'Max variance captured = ' + str(max(cumulative_variance)) )
            

        plot_series(cumulative_variance, threshold = index_of_pca_first_redundant_pca_axis , message = 'PCA: Cumulative Variance')
        
        
        x=reduced_dimention_df[0]
        y=reduced_dimention_df[1]
        
        if n_components == 2:
            plot2d(x,y,color = target, message = message + ' - Top 3 Principle Axis only')
        
        elif n_components > 2:
            z=reduced_dimention_df[2]        
            plot3d(x,y,z,color = target, message = message + ' - Top 3 Principle Axis only')
        
        
        reduced_dimention_df = reduced_dimention_df.drop(columns = list(range(index_of_pca_first_redundant_pca_axis , reduced_dimention_df.shape[1])))
        
        
    elif algorithm == 'cuda_tsne':
        
        reduced_dimention_np_arr = TSNE(n_components=n_components, perplexity=perplexity, learning_rate = learning_rate, verbose=1).fit_transform(df)
        reduced_dimention_df = pd.DataFrame(reduced_dimention_np_arr)
        
        x=reduced_dimention_df[0]
        y=reduced_dimention_df[1]
        
        color = target
        plot2d(x,y, message = 'CUDA TSNE')
        
        plt.scatter(x,y, color)
            
        plt.figure()
        plt.show()
        
            
    elif algorithm == 'isomap':
        reduced_dimention_df = iso_map(df, target,  n_components = n_components)      
    
    else:
        tsne = manifold.TSNE(n_components = n_components, perplexity = perplexity, learning_rate = learning_rate)
        reduced_dimention_np_arr = tsne.fit_transform(X = df)   
        reduced_dimention_df = pd.DataFrame(reduced_dimention_np_arr)
        
        x=reduced_dimention_df[0]
        y=reduced_dimention_df[1]        
        color = target
        
        
        if n_components == 3:
            z=reduced_dimention_df[2]
            plot3d(x,y,z,color, message = message)

        elif n_components == 2:           
           plot2d(x,y,color = color, message = 'TSNE') 
           
    return reduced_dimention_df



def remove_outliers(df, n_estimators = 1000, contamination = .01, message = ''):    
    isoforest = ensemble.IsolationForest(n_estimators = n_estimators, contamination = contamination, n_jobs = -1 )
    outliers = isoforest.fit_predict(df)
    outliers = pd.DataFrame(outliers)
    
    x=df[0]
    y=df[1]
    
    
    if show_plots:
        if df.shape[1] == 2:
            plot2d(x,y,color = outliers, message = message)
        
        elif df.shape[1] > 2:
            z=df[2]        
            plot3d(x,y,z,color = outliers, message = message + ' C='+ str(contamination) +' - Top 3 Axis only')
    
    inliers = df[outliers[0] == 1]
    inliers = pd.DataFrame(inliers)

    return inliers



#def plot_all_continuous_data_kde(df_local):
#    plt.figure()
#    cont_column_names = list(combined_continuous_df.columns)
#
#    for column in cont_column_names:     
#        sns.kdeplot(df_local[column])        
#    plt.show()
#
#
#







use_adaboost = False

if use_adaboost:
    ada_boost_analysis = ada_boost(X_train, y_train)






cf.set_config_file(offline=False, world_readable=False, theme='pearl')
X_train.iplot(kind='box', boxpoints='outliers')




#
message = 'scaled_standard'
X_train = X_train_dict['original'].copy(deep = True)
X_train = standard_scaler(X_train, message, show_plots_local = show_plots)
X_train_dict[message] = X_train




#
message = 'scaled_min_max'
X_train = X_train_dict['original'].copy(deep = True)
X_train = min_max_scaler(X_train, message, show_plots_local = show_plots)
X_train_dict[message] = X_train




#where reduce_dimentions(X_train.iloc[:,:] is used as otherwise it causes the tsne cuda to crash. 
message = 'reduced_dims_on_standard_scaled_pca'
X_train = X_train_dict['scaled_standard'].copy(deep = True)
X_train = reduce_dimentions(X_train.iloc[:,:], y_train, algorithm = 'pca', show_graphs = True, message = message)
X_train_dict[message] = X_train



#where reduce_dimentions(X_train.iloc[:,:] is used as otherwise it causes the tsne cuda to crash. 
message = 'reduced_dims_on_min_max_scaled_pca'
X_train = X_train_dict['scaled_min_max'].copy(deep = True)
X_train = reduce_dimentions(X_train.iloc[:,:], y_train, algorithm = 'pca', show_graphs = True, message = message)
X_train_dict[message] = X_train



# Do thse in loops over all X_train

#message = 'reduced_dims_on_scaled_pca'
#X_train = X_train_dict['scaled_min_max'].copy(deep = True)
#X_train_dict[message] = X_train
#
#
#message = 'reduced_dims_tsne_on_scaled_pca'
#X_train = X_train_dict['reduced_dims_on_scaled_pca'].copy(deep = True)
#X_train = reduce_dimentions(X_train.iloc[:,:], y_train, n_components = 2, algorithm = 'tsne', perplexity = 30, show_graphs = True, learning_rate = 10, message = message)
#X_train_dict[message] = X_train
#
#
#
#message = 'reduced_dims_Isomap'
#X_train = X_train_dict['scaled'].copy(deep = True)
#X_train = reduce_dimentions(X_train, y_train, algorithm = 'isomap', n_components = 3, show_graphs = True, message = message)
#X_train_dict[message] = X_train
#
#
#
#
#message = 'removed_outliers_on_pca_data_with_isolation_forest'
#X_train = X_train_dict['reduced_dims_on_scaled_pca'].copy(deep = True)
#X_train_dict[message] = remove_outliers(X_train, n_estimators = 10000, contamination = .01, message = message )


#
#from tpot import TPOTClassifier
#from tpot import TPOTRegressor
#import tpot
#from joblib import dump
#
#
#memory_dir = os.path.join(directory, 'memory')
#
#
#
#warm_start = False
#
#if not os.path.isdir(memory_dir):
#    os.makedirs(memory_dir)
#    warm_start = False
#else:
#    warm_start = True
#
#
#
#if global_problem_type == 'categorical':
#    pipeline_optimizer = TPOTClassifier(generations=50, population_size=20, cv=5, random_state=42, verbosity=2, scoring = global_scoring,  memory = memory_dir, n_jobs=-1, warm_start = warm_start)
#    
#elif global_problem_type == 'regression':
#    pipeline_optimizer = TPOTRegressor(generations=50, population_size=20, cv=5, random_state=42, verbosity=2, scoring = global_scoring,  memory = memory_dir, n_jobs=-1, warm_start = warm_start)
#     
#
#X_train = X_train_dict['original'].copy(deep = True)
#   
#pipeline_optimizer.fit(X_train, y_train)
#
#
#
##result_file_name = os.path.join(directory, 'tpot_GA_results.py')
#
#results_dir_ga = os.path.join(results_dir,  'Genetic Algorithm Pipeline')  
#ga_generated_code_file = os.path.join(results_dir_ga,  'tpot_GA_results.py')  
#
#if not os.path.isdir(results_dir_ga):
#    os.makedirs(results_dir_ga)
#
#
##saving
#pipeline_optimizer.export(ga_generated_code_file)
#pickle_best_pipeline_file = os.path.join(results_dir_ga , 'pickled_pipeline_object')
#dump(pipeline_optimizer.fitted_pipeline_, pickle_best_pipeline_file)
#
#
#







plt.figure()
plt.title('Target - Normal Distribution')
sns.distplot(y_train, fit=stats.norm, color = 'r', kde = True, rug = True)

#plt.figure(2)
plt.figure()
plt.title('Target - Log Normal Distribution')
sns.distplot(y_train, fit=stats.lognorm, color = 'r', kde = True, rug = False)


#plt.figure(3)
plt.figure()
plt.title('Target - Jhonson Distribution')
sns.distplot(y_train, fit=stats.johnsonsu, color = 'r', kde = True, rug = False)
plt.show()


plt.figure()
plt.title('Target - Cauchy Distribution')
sns.distplot(y_train, fit=stats.cauchy, color = 'r', kde = True, rug = False)
plt.show()


# Get the fitted parameters used by the function
#(mu, sigma) = stats.norm.fit(y_train)
#
## plot with the distribution
#plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
#plt.ylabel('Frequency')

#Probablity plot




from sklearn.preprocessing import PowerTransformer


def analyse_target(y_train_local, message = ''):   
    y_train_local = y_train_dict['original']

    k2, p = stats.normaltest(y_train_local)
    alpha = 1e-3
    print("p = {:g}".format(p))
    
    if p < alpha:  # null hypothesis: x comes from a normal distribution
        print("The null hypothesis can be rejected")
    else:
        print("The null hypothesis cannot be rejected")



    plt.figure()
    stats.probplot(y_train_local, plot=plt)
    title = message + ' - ' + 'Before Transformations' + '.jpg'
    plt.title(title,fontsize=18)        
    file_name = os.path.join(results_dir , title )    
    plt.savefig(file_name, dpi = 1200)

    plt.show()



    transformers = ['yeo-johnson', 'box-cox']

    for transformer in transformers:
      
        y_train_local = y_train_dict['original']

        if transformer == 'box-cox' and y_train_local.min() < 0 :
            #check if all values are positive as this is a requirement for box-cox
            pass

        y_train_local = y_train_local.values
        y_train_local = y_train_local.reshape(-1,1)
        
        pt = PowerTransformer()
        y_train_local = pt.fit_transform(y_train_local)
        y_train_local = np.ndarray.flatten(y_train_local)
        y_train_local = pd.Series(y_train_local)
        y_train_dict[transformer]  = y_train_local
    
    
        #to do: save this using plotter
        plt.figure()
        stats.probplot(y_train_dict[transformer] ,  plot=plt)
        title = message + ' - ' + transformer  + '.jpg'
        plt.title(title,fontsize=18)
        
        file_name = os.path.join(results_dir , title )    
        plt.savefig(file_name, dpi = 1200)

        plt.show()
    
    
analyse_target(y_train_dict, 'Target')    
    


plt.figure()
plt.title('Target - Jhonson Distribution')
y_train = y_train_dict['yeo-johnson']
sns.distplot(y_train, fit=stats.johnsonsu, color = 'r', kde = True, rug = False)
plt.show()

























#s = pickle.dumps(pipeline_optimizer.fitted_pipeline_)


#for perplexity in range(20,10000, 100):
#    message = 'reduced_dims_on_scaled_tsne'
#    X_train = X_train_dict['scaled'].copy(deep = True)
#    X_train = reduce_dimentions(X_train.iloc[:,:], y_train, n_components = 2, algorithm = 'cuda_tsne', perplexity = perplexity, show_graphs = True, learning_rate = 10)
#    X_train_dict[message] = X_train
#
#
#
#





#X_train_dict['reduced_dims_on_unscaled_tsne'].equals(X_train_dict['reduced_dims_on_scaled_tsne'])

#ada_boost_estimator_on_reduced_dims = ada_boost(X_train_dict['scaled'], y_train)
#
#
#ada_boost_estimator_on_reduced_dims = ada_boost(X_train, y_train, 'scaled')
#



#to do: ensure data is scaled







#
#
#message = 'reduced_dims_on_unscaled_tsne'
#X_train = X_train_dict['original'].copy(deep = True)
#X_train = reduce_dimentions(X_train.iloc[:,:], y_train, n_components = 3, algorithm = 'tsne_cuda', perplexity = 30, show_graphs = True, message = message)
#X_train_dict[message] = X_train

#message = 'reduced_dims_on_scaled_tsne_cuda'
#X_train = X_train_dict['scaled'].copy(deep = True)
#X_train = reduce_dimentions(X_train.iloc[:,:], y_train, n_components = 2, algorithm = 'tsne_cuda', perplexity = 30, show_graphs = True, learning_rate = 10, message = message)
#X_train_dict[message] = X_train


#message = 'reduced_dims_on_scaled_tsne'
#X_train = X_train_dict['scaled'].copy(deep = True)
#X_train = reduce_dimentions(X_train.iloc[:,:], y_train, n_components = 2, algorithm = 'tsne', perplexity = 30, show_graphs = True, learning_rate = 10, message = message)
#X_train_dict[message] = X_train












#
#def label_and_one_hot_encode(series):
##    print(series)
#    series = series.tolist()
#    
#    le = preprocessing.LabelEncoder()
#    le.fit(series)
#    series = le.transform(series)
#    
#    ohe = preprocessing.OneHotEncoder(categories='auto')
#    length = len(series)
#    series = series.reshape(length, 1)
#
#    ohe.fit(series)
#    series = ohe.transform(series)
#    return series
#
#combined_categorical_df = combined_categorical_df.apply(label_and_one_hot_encode)
#
#
















































#
#
#def convert_non_to_unknown(column_series):
#    if not column_series.shape[0] == y_train.shape[0]:
#        

#
#for column_name in combined_categorical_df.columns:
#    print(column_name)    
# 
#    column_series = combined_categorical_df[column_name]
#    
#    #label encode if it is a string string
#    if not column_series.dtype.kind in 'biufc':
#       le = LabelEncoder()
#       column_series = le.fit(column_series)
#       column_series = le.transform([column_series])
#       
#    print(column_series)
#


#
#combined_categorical_df = combined_categorical_df.replace(np.NaN, '')
#
#
#
#
#
#column_series = [0,1,2,3,4,5,6, np.NaN]
#
#le = LabelEncoder()
#le.fit(column_series)
#column_series = le.transform(column_series)
#
#
#
#
#
#


        
#def drop_and_log_multi_columns_by_indices(df, indices = None, reason = None): 
#    df_dropped = df.iloc[:,indices].copy(deep = True)
#    df_dropped = df_dropped.add_suffix('_' + reason)
#    
#    auto_generated_data_df_dropped = auto_generated_data_df_dropped.join(df_dropped, how = 'right')
#    
#    df.drop(index = indices, inplace = True)






