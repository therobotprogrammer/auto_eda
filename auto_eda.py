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
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import keras
import math
import mpu
import string
import re
#from sklearn_pandas import CategoricalImputer



train = pd.read_csv('/home/pt/Documents/auto_eda/train.csv', index_col = False)


train.info()
train.describe()

max_groups_in_categorical_data = 25
min_members_in_a_group = 5

auto_min_entries_in_continous_column = 10
hashing_dimention = 2


target_column = 'Survived'

target_series = pd.Series()

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


def hash_it(text):
    if (text == np.nan) or (text == None) or (text == 'nan') or type(text) == type(np.nan) :
       return np.nan
    text = str(text)
    hash_list = keras.preprocessing.text.hashing_trick(text, hashing_dimention, hash_function=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')

    #convert list of hashes to a single hash. This is because keras hashes multiple works together and returns their hashes 
    #as a list. But in this case we only have a single hash. So the list is collapsed and then converted to a string
    hash_str = ''.join(str(e) for e in hash_list)
    
    #convert hash string to int
    hash_int = int(hash_str)    
                                                     
    return(hash_int)   
                                              

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
                pass            
        
                        
            if column != target_column:
                print(log_column_str, 'vs target ', log_target_str)
                plt.figure()
                sns.catplot(x= column, hue = target_column, data = train, kind = 'count', height=6)
                plt.show()
#                sns.factorplot(x="Embarked", hue="Survived", data=titanic_train, kind="count", height=6)
                train_categorical[column] = temp_column
                
                
            else:
                target_series = temp_column        
                
                
        else:    
            if train[column].dtypes != 'O' and column != target_column:        
                try:
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

        #pd.concat(auto_generated_categories, tags)
        auto_generated_data_df = auto_generated_data_df.join(multi_column_df, how = 'outer')




print()
print('Finding possible groups in cleaned data')


 
auto_generated_data_df.columns
#auto_generated_data_df_raw = auto_generated_data_df
auto_generated_data_df_categorical = auto_generated_data_df.copy(deep = True)

#this is only to generate graphs. seaborn needs same dataframe for faced grids
#auto_generated_data_df[target_column] = train[target_column]

auto_generated_hash_df = pd.DataFrame()
auto_generated_data_df_continuous = pd.DataFrame()

auto_generated_data_df_dropped = pd.DataFrame()



#convert possible numbers to numeric type
for column in auto_generated_data_df.columns:
    #try to convert columns with all numbers to integers
    try:
        auto_generated_data_df[column] = pd.to_numeric(auto_generated_data_df[column])
        print('Converted auto generated column to int: ', column)
    except Exception:
        pass     


for column in auto_generated_data_df.columns:
        
    if column == target_column:
        continue
   
    found_new_feature = False
    
    group_obj = auto_generated_data_df.groupby(column)
    group_dict = group_obj.groups
    group_names = group_dict.keys()   
    
    total_groups = group_obj.size().shape[0]
    
    
    column_str = '[' + column +']'    
#    if column == 'Ticket_2':
#        print('Type: ', type(auto_generated_hash_df[column] ))

    
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
            else:
                #mark all members with few members as discarded
                rows_to_discard = group_dict[key]
                auto_generated_data_df_categorical.loc[rows_to_discard, column] = np.nan                
    
    ### To Do: Plot graphs later        
    if found_new_feature:                
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
                auto_generated_hash_df[column] = auto_generated_hash_df[column].apply(hash_it)            
                pass

        else:
            print(column_str, 'Too little data for continuous')
            auto_generated_data_df_dropped = auto_generated_data_df[column]
            
            
#To Do: These need to be added to auto_generated_data_df_dropped
print('Dropping auto generated columns with no features')
auto_generated_data_df_categorical = auto_generated_data_df_categorical.dropna(axis='columns', how = 'all')            
            
#            sns.FacetGrid(train, hue = "Survived", height=6).map(sns.distplot, 'temp', kde = True).add_legend()             
#            sns.FacetGrid(train, hue = "Survived", height=6).map(sns.distplot, 'temp', kde = False).add_legend()  



            


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
        ax = sns.countplot(auto_generated_hash_df[column], hue = train[target_column])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
        plt.tight_layout()
        plt.show()           
    except Exception:
        pass  
                        
#
#auto_generated_hash_df['Sex'] = train['Sex']
#auto_generated_hash_df['Survived'] = train['Survived']
#
#
#g = sns.FacetGrid(auto_generated_hash_df, row="Name_5", col="Survived") 
#g.map(sns.countplot, "Sex")
#
#g = sns.FacetGrid(auto_generated_hash_df, row="Name_5", col="Sex") 
#g.map(sns.countplot, "Survived")
#
#g = sns.FacetGrid(auto_generated_hash_df, row="Cabin_3", col="Survived") 
#g.map(sns.countplot, "Sex")

#TO DO: GET TICKET_0 IN HASH





combined_categorical_df = pd.DataFrame()
combined_categorical_df = train_categorical.join(auto_generated_data_df_categorical,how = 'outer')
combined_categorical_df = combined_categorical_df.join( auto_generated_hash_df, how = 'outer')

from sklearn_pandas import CategoricalImputer
from sklearn.preprocessing import OneHotEncoder


titanic_train = train


#ohe_features = ['Sex','Embarked','Pclass']
#ohe = preprocessing.OneHotEncoder()
#ohe.fit(titanic_train[ohe_features])
#print(ohe.n_values_)
#tmp1 = ohe.transform(titanic_train[ohe_features]).toarray()
#

#
#lab_encoder_sex = preprocessing.LabelEncoder()
#lab_encoder_sex.fit(titanic_train['Pclass'])
#print(lab_encoder_sex.classes_)
#titanic_train['PassengerId'] = lab_encoder_sex.transform(titanic_train['PassengerId'])
#


 
#
#def impute_cat_column(series):
#    print(series)    
#
#    if not series.dtype.kind in 'biufc':  
#        series = series.replace(np.NaN, 'Unknown')        
#        
#    le = LabelEncoder()
#    le.fit(series)
#    series = le.transform(series)
#    return series
#    
#combined_categorical_processed_df = pd.DataFrame()
#
#combined_categorical_processed_df = combined_categorical_df.apply(impute_cat_column)


























































#
#
#def convert_non_to_unknown(column_series):
#    if not column_series.shape[0] == target_series.shape[0]:
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





