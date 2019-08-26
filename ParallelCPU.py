#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 14:46:12 2019

@author: pt
"""



import multiprocessing
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from funcy import join
import math






#class DfTransformer:    
#    def __init__(self):
#        self.df_original_info = {}
#        self.df_list = {}
#
#
#    def transpose(self, original_df):    
#        for column in original_df.columns:
#            data_type = str(original_df[column].dtype)            
#            self.df_original_info_dict[column] = data_type    
#        return original_df.transpose
#    
#    
#    def revert_dtypes(self, single_transposed_df):
#        double_transposed_df = single_transposed_df.transpose()
#        
#        for column in double_transposed_df.columns:
#
#            dtype = self.df_original_info[column]
#            double_transposed_df[column].astype(dtype)
#            
#            return double_transposed_df
        

#class DataFrameJoiner:
#    def __init__(self, df_list = []):
#        self.df_list = df_list
#        self.dtypes = {}
#        
#    def append(self, sub_df):
#        self.df_list.append(sub_df)
#      
#        
#    def __get_dtype_dict(self):
#        for df in self.df_list:
#            for column in df.columns:
#                data_type = str(df[column].dtype)       
#                print('Column:' , column , '   Dtype:', data_type )
#
#                self.dtypes[column] = data_type
#    
#    def __revert_original_column_types(self, joined_data_df):    
#        for column in joined_data_df.columns:
#            dtype = self.dtypes[column]
#            joined_data_df[column].astype(dtype)
#            
#        return joined_data_df
#            
#    def get_joined_data_df(self):        
#
#        self.__get_dtype_dict()
#        
#        for idx, sub_df in enumerate(self.df_list):
#            self.df_list[idx] = sub_df.T
#
#        joined_data_df = pd.concat(self.df_list, axis = 0)      
#        joined_data_df = joined_data_df.T 
#        
#        joined_data_df = self.__revert_original_column_types(joined_data_df)
#            
#        
#        return joined_data_df        
            
            
 

def join_dataframes(df_list):

    dtypes = {}
    
    
    count_df_with_common_columns = 0
    
    for df in df_list:
        if all(df_list[0].columns == df.columns) :
            count_df_with_common_columns +=1
        else:
            break
        
    if count_df_with_common_columns == len(df_list):
        joined_data_df = pd.concat(df_list, axis = 0)   
        return joined_data_df
    
    else:
        #preserve datatypes before transpose
#        for df in df_list:
#            for column in df.columns:
#                data_type = str(df[column].dtype)       
#    #            print('Column:' , column , '   Dtype:', data_type )
#    
#                dtypes[column] = data_type    
#    
#        #rotate individual dataframes
#        for idx, sub_df in enumerate(df_list):
#            df_list[idx] = sub_df.T
    
        #combine dataframes
        joined_data_df = pd.concat(df_list, axis = 1)      
#        joined_data_df = joined_data_df.T 
        
        #revert original column datatypes that are lost due to transpose
#        for column in joined_data_df.columns:
#            dtype = dtypes[column]
#            joined_data_df[column] = joined_data_df[column].astype(dtype)
            
        return joined_data_df    



class ParallelCPU:

    def __init__(self,   n_cores = -1, verify_parallel_execution = False, debug_mode = False):
        if n_cores == -1 or n_cores == 0:
            self.n_cores = multiprocessing.cpu_count() 
        self.verify_parallel_execution = verify_parallel_execution
        self.debug_mode = debug_mode
  
        
    def compute(self, df_input, function):
        # calculate features in paralell by splitting the dataframe into partitions and using paralell processes
        processes = self.n_cores
          
        if self.debug_mode == True:
            compiled_result_serial = function(df_input)
            return compiled_result_serial
    
        else:
                total_columns = df_input.columns.shape[0]
                
                max_columns_per_core = total_columns/processes
                max_columns_per_core = math.ceil(max_columns_per_core)
                
                df_split = []
#                df_split = np.array_split(df_input, partitions, axis=1)  # split dataframe into partitions column wise
#                
                for idx in list(range(0, total_columns, max_columns_per_core)):
                    col_min = idx
                    col_max = idx + max_columns_per_core 
                    
#                    print(col_min , ' ' , col_max)
                    
                    sub_df = df_input.iloc[:, col_min: col_max]
                    
                    df_split.append(sub_df)
                
                
                with ProcessPoolExecutor(processes) as pool:     
                    results_generator = pool.map(function, df_split)             
                    results_as_splits = list(results_generator)
               
                compiled_result_parallel = [] 
                
                if isinstance(results_as_splits[0], tuple):                
                    
                    items_per_split =  len(results_as_splits[0])                      
                    
                    
                    
                    for index in range(0, items_per_split):
                        splits_to_concatenate = []
                        
                        
                        for split in results_as_splits:
                            
                            #If multiple values are returned, then they are a tuple. They have to be accessed as index. 
                            if isinstance(split[index], pd.DataFrame) :
                                splits_to_concatenate.append(split[index]) 
                             
                            else:
                                splits_to_concatenate.append(split[index])                            
                       
                        #concatenate based on datatype
                        if isinstance(splits_to_concatenate[0], pd.DataFrame):
                            joined_data_df = join_dataframes(splits_to_concatenate)
#                            original_dtypes = splits_to_concatenate[0].
#                            joined_data_df = pd.concat(splits_to_concatenate, axis = 0) 
#                            joined_data_df = joined_data_df.T #Refer comment
                            
                            compiled_result_parallel.append(joined_data_df)                        
                       
                        else:
                            joined_data = join(splits_to_concatenate)
                            compiled_result_parallel.append(joined_data)
                                
                    compiled_result_parallel = tuple(compiled_result_parallel)
     
        
                else:
                        # If a tuple was not returned 
                        splits_to_concatenate = []
                        
                        for split in results_as_splits:
#                            #problem    
#                            splits_to_concatenate.append(split)                            
                       
                            if isinstance(split, pd.DataFrame) :
                                splits_to_concatenate.append(split) 
                             
                            else:
                                splits_to_concatenate.append(split) 
                                
                                
                        #concatenate based on datatype
                        if isinstance(splits_to_concatenate[0], pd.DataFrame):
                            joined_data_df = join_dataframes(splits_to_concatenate)
                            compiled_result_parallel.append(joined_data_df)    
                            
                        elif isinstance(splits_to_concatenate[0], type(None)):
                            compiled_result_parallel.append(None)
                            
                        else:
                            joined_data = join(splits_to_concatenate)
                            compiled_result_parallel.append(joined_data)
                            
                        #since original function returned only one item, 
                        #this function also returns only 1 item and not a tuple of items
                        
                        compiled_result_parallel = compiled_result_parallel[0]      
        
                if self.verify_parallel_execution == True:
                    compiled_result_serial = function(df_input)
    
    
                    if isinstance(compiled_result_serial, tuple):
                        for index, serial_item in enumerate(compiled_result_serial) :
                            if isinstance(serial_item, pd.DataFrame) :
                                parallel_item = compiled_result_parallel[index]
                                assert serial_item.equals(parallel_item), 'Serial and Parallel compution do not match'
                            
                            else:
                                assert serial_item == compiled_result_parallel[index] , 'Serial and Parallel compution do not match'
                                
                    else:
                            if isinstance(compiled_result_serial, pd.DataFrame) :
                                assert compiled_result_serial.equals( compiled_result_parallel ), 'Serial and Parallel compution do not match'
                            
                            else:
                                assert compiled_result_serial == compiled_result_parallel , 'Serial and Parallel compution do not match'                     
        
        return compiled_result_parallel









if __name__ == '__main__':
 
    from numbers import Number
       
    
    def sqrt(df):    
        result = pd.DataFrame()
        
        for idx, column in enumerate(df.columns):
            result[column] =  np.sqrt( df[column] ) 
                
        return result


    
    def square(df):
        result = pd.DataFrame()
        
        for idx, column in enumerate(df.columns):
            result[column] =  np.square(df[column]) 
                    
        return result    
    
    
    
    
    
    def square_numbers_and_text(df):
        result = pd.DataFrame()
        result_dict = {}
        
        
        for idx, column in enumerate(df.columns):
            if isinstance(df[column][0] , Number):
                result[column] =  np.square(df[column]) 
            else:
                result[column] =  df[column]
                
            group_obj= df.groupby(by = column)
            groups_dict = group_obj.groups
            
            for key in groups_dict.keys():
                result_dict[key] = groups_dict[key].values
            
        return result, result_dict
    
    
        
    
    parallel = ParallelCPU(verify_parallel_execution = False, debug_mode = False)
    
    
    df = pd.DataFrame()   
    
    df['A'] = list(range(0,10))
    df['B'] = df['A'] * 2
    df['C'] = df['A'] * 3
    df['D'] = df['A'] * 4

    df_squared_serial = df.apply(np.square)    
    df_squared_parallel = parallel.compute(df, square)      
    assert(df_squared_serial.equals(df_squared_parallel))
    
    
    df_sqrt_serial = df_squared_serial.apply(np.sqrt)    
    df_sqrt_parallel = parallel.compute(df_squared_parallel, np.sqrt)        
    assert(df_sqrt_serial.equals(df_sqrt_parallel))
    
    
    df['Names'] = [
                    'a',
                    'b',
                    'c',
                    'd',
                    'e',
                    'f',
                    'a',
                    'b',
                    'a',  
                    'a',
                    ]    
    
    
    df['Names_2'] = df['Names']*2 
    df['Names_3'] = df['Names']*3 
    df['Names_4'] = df['Names']*4 
    
    
    
    
    df_serial, list_of_group_dicts_serial = square_numbers_and_text(df)    
    df_parallel, list_of_group_dicts_parallel = parallel.compute(df, square_numbers_and_text)      
    assert(df_serial.equals(df_parallel))
    
    
    df_serial, dicts_serial = square_numbers_and_text(df)    
    df_parallel, dicts_parallel = parallel.compute(df, square_numbers_and_text)      

    print('Everything worked sucessfully.')
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

