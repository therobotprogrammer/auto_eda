#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:26:38 2019

@author: pt
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 16:37:03 2019

@author: pt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing, neighbors, decomposition
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


import plotly 

import plotly.plotly as py
import plotly.graph_objs as go


from missingpy import KNNImputer, MissForest



##############################################
# ALL TO DO
# try without using missing data
# to do: change hash function as md5 data to be hashed randomly. As a result the rf imputer throws it. 
# to do: change hash function as md5 data to be hashed randomly. As a result the rf imputer throws it.  


auto_generated_data_df_dropped = pd.DataFrame()

log_warnings = []


#from sklearn_pandas import CategoricalImputer

### To Do: Convert comma to space

directory = '/home/pt/Documents/auto_eda'

file_path = os.path.join(directory, 'train.csv')
train = pd.read_csv(file_path, index_col = False)

file_path = os.path.join(directory, 'pass_nationalities.csv')
name_prism = pd.read_csv(file_path, index_col = False)
name_prism_train = name_prism[:train.shape[0]]

train['nationality'] = name_prism_train['Nationality']

show_plots = True


train.info()
train.describe()

max_groups_in_categorical_data = 50
min_members_in_a_group = 5

auto_min_entries_in_continous_column = 10
hashing_dimention = 256
min_data_in_column_percent = 0 #percent of nan . To Do: This should also be applied to continuous

#to do: hash char as a to z and int as a range
#to do: for things that are hashed, use box plot etc

target_column = 'Survived'

y_train = pd.Series()

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
    #convert to lower case
    s_lower = s.lower()
    
    return ord(s_lower[0])
    
    

def hash_it(text, hashing_strategy = 'ascii_0'):
    
    if hashing_strategy == 'md5':
        hash_function = 'md5'
        n = 2
    elif hashing_strategy == 'ascii_0':
        hash_function = hash_function_first_char
        n = 256
    else:
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
        
    


def label_encoder(df_original, strategy = 'keep_missing_as_nan'):    
    
    assert (min_data_in_column_percent >= 0 and min_data_in_column_percent <= 1), 'min_data_in_column_percent out of range 0 to 1'
        
    
    df = df_original.copy(deep = True)
    
    mask_df = pd.DataFrame(df.isnull().values)
    
#    mask_df.columns = df.columns
        
    sys_min = -sys.maxsize -1   
   
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
    log_warnings.append(w)
    print(w)
    
    


def drop_and_log_column(df, column_name = None, reason = None):
    auto_generated_data_df_dropped[column_name + '__' + reason] = df[column_name].copy(deep=True)
    df.drop(columns = column_name, inplace = True)
    

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
    file_path = os.path.join(os.getcwd(), filename)
    
    export_graphviz(dt, out_file=dot_data,  
                    filled=True, rounded=True,
                    feature_names = X_train.columns, class_names = y_train.name)
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

































column_properties_df = pd.DataFrame(columns = ['categorical', 'text', 'unresolved', 'incomplete', 'imputed', 'error'])

train_categorical = pd.DataFrame()
train_continuous = pd.DataFrame()
 

for idx, column in enumerate(train.columns):
    print()
    try:
        #column_properties_df.loc[column, 'Index'] = column
        column_properties_df.loc[column, 'error'] = False
        group_obj = train.groupby(column)
        group_count_in_column = group_obj.count().shape[0]
        
        categorical = False
        
        log_column_str = 'Log: >>>>> ' + '[' +  column + '] '
        log_target_str = '[' + target_column +']'
        
        if group_count_in_column <= max_groups_in_categorical_data:
            print('Log: [', column,'] is categorical. Unique groups in category: ', group_count_in_column)
            categorical = True
            column_properties_df.loc[column, 'categorical'] = True
            
        if categorical:
#            if train[column].dtypes == 'O':
#                print(log_column_str, ' Type is Object. Applying label encoding ')
#                labelencoder = LabelEncoder()
#                labelencoder.fit_transform(train[column])
#                train[column] = labelencoder.transform(train[column])
                     
            #try to convert data to numeric if possible
            try:
                temp_column = pd.to_numeric(train[column])                
            except Exception:
                temp_column = train[column]
                pass            
        
                        
            if column != target_column:
                if show_plots:
                    print(log_column_str, 'vs target ', log_target_str)
                    plt.figure()
                    sns.catplot(x= column, hue = target_column, data = train, kind = 'count', height=6)
                    plt.show()
#                sns.factorplot(x="Embarked", hue="Survived", data=titanic_train, kind="count", height=6)
                train_categorical[column] = temp_column
                
                
            else:
                y_train = temp_column        
                
                
        else:    
            if train[column].dtypes != 'O' and column != target_column:        
                try:
                        if show_plots:
                            plt.figure()
                            sns.FacetGrid(train, hue = "Survived", height=6).map(sns.kdeplot, column).add_legend()  
                            plt.show()
                        column_properties_df.loc[column, 'categorical'] = False
                        
                        train_continuous[column] = train[column]


                except:
                        print('Log: >>>>> Unknown error: Cannot make graph for column: ', column)
                        column_properties_df.loc[column, 'error'] = True

            else:
                print(log_column_str , 'Cannot make graph for continous column and object. Will be processed as text later')
                column_properties_df.loc[column, 'unresolved'] = True


    except:
        print(log_column_str , 'Preprocessing failed. Proceeding to next column')
            
#print('Log: ' , unresolved_columns, 'Could not be resolved as they were continous and of type object. These will be taken as string')       
plt.show()



print()
print()
print('Cleaning text in unresolved columns')



column_split_mapper_dict = {}


# Clean text columns
auto_generated_data_df = pd.DataFrame()
for index, row in column_properties_df.iterrows():
    list_of_autogenerated_df = []
    
    column_name_in_train_df = index
    if row['unresolved'] == True:
        print(index)        
        # first split with space, then remove special charecters
#        cleaned_and_split_df  = train[column_name_in_train_df].apply(clean_and_split_text)
        
        cleaned_and_split_df  = train[column_name_in_train_df].apply(clean_text, filters = '\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
                                              
        cleaned_and_split_df  = cleaned_and_split_df.apply(split_text, split_at = ' ')  
        
        cleaned_and_split_df = cleaned_and_split_df.apply(split_at_char_int_transistion)
        

            
        
        #convert column of lists into its own seperate column
        multi_column_df = cleaned_and_split_df.apply(pd.Series)

        #rename columnn names to include original name
        multi_column_df = multi_column_df.rename(columns = lambda x : column_name_in_train_df + '_' + str(x))
        
        column_split_mapper_dict[column_name_in_train_df] = list(multi_column_df.columns)

        #pd.concat(auto_generated_categories, tags)
        auto_generated_data_df = auto_generated_data_df.join(multi_column_df, how = 'outer')




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
for column in auto_generated_data_df.columns:
    #try to convert columns with all numbers to integers
    for row in auto_generated_data_df[column]:

        try:
            auto_generated_data_df.iloc[row,column] = pd.to_numeric(auto_generated_data_df.iloc[row,column])
            
            print('Converted auto generated column to int: ', column)
        except Exception:
            pass     



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
                plt.figure()
                ax = sns.countplot(auto_generated_data_df_categorical[column], hue = train[target_column])
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
                plt.tight_layout()
                plt.show()
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
                auto_generated_hash_df[column] = auto_generated_hash_df[column].apply(hash_it, hashing_strategy = 'md5')   
                
                
                
                
                pass

        else:
            print(column_str, 'Too little data for continuous')
            auto_generated_data_df_dropped = auto_generated_data_df[column]
            
            
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
            sns.FacetGrid(auto_generated_data_df_continuous, hue = target_column, height=6).map(sns.distplot, column, kde = True).add_legend()             
            sns.FacetGrid(auto_generated_data_df_continuous, hue = target_column, height=6).map(sns.distplot, column, kde = False).add_legend()     
           
            plt.tight_layout()
            plt.show()        
    
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
    except Exception:
        pass  
                        

def pre_processing(categorical_df, continuous_df, imputer, enable_ohe, exclude_column_from_ohe):   
    
    
    #label encode categorical
    
    categorical_df_label_encoded =  label_encoder(categorical_df, strategy = 'keep_missing_as_nan')
    #here outer is used. so continuous
    cat_columns = categorical_df_label_encoded.columns
    cont_columns = continuous_df.columns
    
    joint_df = categorical_df_label_encoded.join( continuous_df, how = 'outer')      

    #get indexes of categorical columns. this is used by miss_forest to identify categorical features
    cat_column_indexes_in_joint_df = []    
    for column in cat_columns:
        cat_column_indexes_in_joint_df.append(joint_df.columns.get_loc(column))
    
    
    cat_column_indexes_in_joint_df = np.asarray(cat_column_indexes_in_joint_df)
    
    
    if imputer == 'random_forest':
        print('Using random forest as imputer')
        imp = MissForest(max_iter = 20, n_estimators = 1000, n_jobs = 24, verbose = 0)
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
    
    if enable_ohe:
        #the cat and cont are seperated and then joint so that the order is preserved after one hot
        imputed_cat_df = imputed_df[list(cat_columns)]
        imputed_cont_df = imputed_df[list(cont_columns)]
        
        
        #one hot categorical columns    
        imputed_cat_df = one_hot(imputed_cat_df, exclude_column_from_ohe)
        
        imputed_df = imputed_cat_df.join(imputed_cont_df,how = 'outer')

    
    
    return imputed_df
    

def one_hot(df, exclude_from_ohe = []):
    cat_columns = df.columns
    
    columns_to_one_hot = []
    
    df_excluded_from_one_hot = pd.DataFrame()
    
    df_after_one_hot = pd.DataFrame()

    if len(exclude_from_ohe) == 0:
        df_after_one_hot = pd.get_dummies(df, columns= columns_to_one_hot )
    
    else:         
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

combined_categorical_df = pd.DataFrame()
combined_categorical_df = combined_categorical_df.join(train_categorical,how = 'outer')
combined_categorical_df = combined_categorical_df.join(auto_generated_data_df_categorical,how = 'outer')
combined_categorical_df = combined_categorical_df.join( auto_generated_hash_df, how = 'outer')




combined_continuous_df = pd.DataFrame()
combined_continuous_df = combined_continuous_df.join(train_continuous,how = 'outer')
combined_continuous_df = combined_continuous_df.join(auto_generated_data_df_continuous,how = 'outer')

exclude_from_ohe = ['Pclass', 'SibSp', 'Parch' ]

X_train = pre_processing(combined_categorical_df, combined_continuous_df, imputer = 'random_forest', enable_ohe = True, exclude_column_from_ohe = exclude_from_ohe)


#combined_categorical_preprocessed_df = cat_preprocess(combined_categorical_df, exclude_from_ohe = exclude_from_ohe)


#def cast_cont_to_cat(df, features):
#    for feature in df.columns:
#        df[feature] = df[feature].astype('category')
#        
#        


#combined_continuous_preprocessed_df = cont_preprocess(combined_continuous_df)

#warning: turning off imputation

#turn_off_cont_preprocessing = False
#
#if turn_off_cont_preprocessing:
#    print('Warning: Continuous Features Preprocessing including Imputation turned off')
#    combined_continuous_preprocessed_df = combined_continuous_df
#else:
    

#combined_continuous_preprocessed_df = cont_preprocess(combined_continuous_df, missing_data_drop_threshold = .5, si_strategy = 'constant', si_fill_value = -123456)
    


#drop columns
# passenger id is dropped as it uniquely identifies the row. This causes overfitting when a lot of trees are used. The model
# remembers the passenger id and result

drop_and_log_column(X_train, 'PassengerId', 'manually dropped - unique identifier')
#
#
#tree
dtree = tree.DecisionTreeClassifier()
dtree.fit(X_train, y_train)

dtree_pre_analysis = analyze_tree(dtree, X_train, y_train, max_depth_search_list_length = 10)


####################################################################################################3

use_dtree = False
if use_dtree:
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
    
def plot_3d(grid_estimator, plot_type = 'mesh'):
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
                       colorscale="Rainbow",
                       intensity=train_scores,                        
                       )
            
            trace_test = go.Mesh3d(x=x,y=y,z=test_scores,
                       alphahull=3,
                       opacity=.5,
                       colorscale="Picnic",
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



use_ada_boost = True
if use_ada_boost:    
    base_classifier = tree.DecisionTreeClassifier()
    
    ada_boost_dt = ensemble.AdaBoostClassifier(base_classifier)
    
    # note: it is very important to specify max depth = 1 so only stumps are built. 
    # if not then whole tree is built causing overfitting. 
    
    ada_boost_grid = {
                        'base_estimator__max_depth' : [1], 
                        'n_estimators': get_equally_spaced_numbers_in_range(1,10000,30) ,
                        'learning_rate': get_equally_spaced_non_zero_floats_in_range(0,1,40) ,
            }
    
    ada_boost_grid_estimator = model_selection.GridSearchCV(ada_boost_dt, ada_boost_grid, scoring = 'accuracy', n_jobs = -1, refit = True, verbose = 1, return_train_score = True, cv =10)
    
    
    ada_boost_grid_estimator.fit(X_train, y_train)
    
    print('Best SCore: ', ada_boost_grid_estimator.best_score_)
    print('Best Params: ', ada_boost_grid_estimator.best_params_)
    
    plot_3d(ada_boost_grid_estimator)
    
    results_ada_boost_grid_estimator = ada_boost_grid_estimator.cv_results_





#####
#gradient boost
#stacking
#voting


#####################
#PCA

def pca(df, target, axis = 8, show_graphs = True):
    lpca = decomposition.PCA(n_components = axis)
    
    lpca.fit(df)
    
    
    df_pca_new_axis = lpca.fit_transform(df)
    df_pca_original_axis = lpca.inverse_transform(df_pca_new_axis)
    
    df_pca_new_axis_df = pd.DataFrame(df_pca_new_axis)
    #df_pca_original_axis_df = pd.DataFrame(df_pca_new_axis)
    
    del(df_pca_new_axis)
    del(df_pca_original_axis)
    
    lpca.explained_variance_ratio_
    
    plt.figure()
    plt.plot(lpca.explained_variance_ratio_)
    plt.ylabel('Variance Ratio')
    plt.xlabel('PCA Axis')
    plt.show()
    
    trace_pca = go.Scatter3d(
        x=df_pca_new_axis_df[0],
        y=df_pca_new_axis_df[1],
        z=df_pca_new_axis_df[2],
        mode='markers',
        marker=dict(
            size=12,
                    # set color to an array/list of desired values
            color=target,
            colorscale='Viridis',   # choose a colorscale
            opacity=0.8
        )
    )
        
    traces = [trace_pca]
           
    plotly.offline.plot(traces)

    return df_pca_new_axis_df



X_train_pca = pca(X_train, y_train, show_graphs = True)






































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






