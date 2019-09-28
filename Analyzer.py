#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 11:51:14 2019

@author: pt
"""
#import pandas as pd
import dask.dataframe as pd




from sklearn.pipeline import make_pipeline, Pipeline
import itertools
import numpy as np
import numbers    
from pandas.api.types import is_numeric_dtype
import re, mpu
import plotly.graph_objects as go
import colorlover as cl


#import deep_compare
import copy

#from deep_eq import deep_eq
import math

from dask_ml.wrappers import ParallelPostFit

from dask_ml.model_selection import GridSearchCV
#from sklearn.model_selection import GridSearchCV


import sys
sys.path.insert(0, '/home/pt/Documents/auto_eda')  
from Plotter import Plotter
from SaveAndLoad import SaveAndLoad
from ParallelCPU import ParallelCPU


import os


    
class Analyzer:
    def __init__(self, message, plot_dir = '', show_plots = True, database = None, debug_mode = False, use_precomputed_results = True, compact_plots = True, use_dask = False, prefix_for_parameters = ''):        
        self.config = None
        self.pipeline = None
        self.lower_is_better = False  
        self.scoring = None
        self.show_plots = show_plots
        self.message = message
        self.use_precomputed_results = use_precomputed_results
        self.compact_plots = compact_plots
        self.use_dask = use_dask
        self.prefix_for_parameters = prefix_for_parameters
        
        if use_precomputed_results == True:
            assert(database is not None)
        
        
        database_top_directory = database.save_directory
        database_sub_directory = os.path.join(database_top_directory, 'Analyzer', message)        
        self.database = SaveAndLoad(database_sub_directory)

        if plot_dir == '':            
            plot_dir = os.getcwd()
        self.plot_dir = plot_dir 
        self.plotter = Plotter(self.plot_dir)        


        self.parallel = ParallelCPU(debug_mode = debug_mode)
        
        self.processing_time_ = 0        
        self.grid_search_arguments = None
        self.cv_results = None
        self.grid_search_estimator = None
        self.processed_results_dict = None
        
        
                

                    
                    
    def gererate_params(self, config_dict):        
        generated_params = self.__dp(config_dict, self.prefix_for_parameters)
        return generated_params
    

    
    def gridsearchcv(self, pipeline, config, **kwargs ):
        self.grid_search_arguments = kwargs        
        self.pipeline = pipeline
        self.grid_search_params = self.gererate_params(config) 


    import inspect


    #this function is needed because math.isnan() will raise an error if it is given a function
    #sometimes we have to check for functions. 
    def check_for_nan(self, v1,v2):
        value_to_return = False
        
        try:
            if (math.isnan(v1)) and (math.isnan(v2)):
                value_to_return = True

        except Exception:
            pass                 
            
        return value_to_return
        
    def __compare_two_params(self, param1, param2):
        if len(param1.keys()) != len(param2.keys()) :
            return False
              
        match = False
        for key, value in param1.items():            
            if key in param2:
                v1 = param1[key][0]
                v2 = param2[key][0]
                
#                print('>>>>>v1: ' , v1)
#                print('>>>>>v2: ' , v2)
                #converting to string as some arguments are functions or nan
                if v1 == v2:
                    continue
                elif ('sklearn' in str(type(v1)) ) and ('sklearn' in str(type(v1)) ) and (type(v1) == type(v2) ):
                    #it is a function of sklearn that matches. 
                    #this is a workaround, because when saving xgboost(), the missing=None becomes missing=nan within this function. 
                    #this causes the equality to fail. this is a workaround.
                    continue

                elif v1 is None and v2 is None:
                    continue
                
                elif self.check_for_nan(v1,v2):
                    continue
                
                else:
                    return False
            else:
                return False   
        return True        
                
            
    
    def load_precomputed_results(self, X, y):
        precomputed_result_archive = self.database.load('precomputed_result_archive')

        reduced_grid_search_params = []
        self.precomputed_result = None 
        self.params_to_load_from_memory = None
        self.precomputed_result_found = False
        self.precomputed_archive_id = None
        
        
        
        self.params_skipped_computation = None
        self.processed_results_dict_skipped_computation = {}
        
        
        matching_precomputed_result = {}

        
        if precomputed_result_archive is not None: 
            for archive_id, result in enumerate(precomputed_result_archive):                
                if result['X'].equals(X) and result['y'].equals(y):
                    self.precomputed_result = result
                    self.precomputed_result_found = True
                    self.precomputed_archive_id = archive_id
                    
                    matching_pre_computed_idx = []
                    matching_pre_computed_params = []
                    matching_pre_computed_processed_results_dict = {}


                    for idx_grid_search_params, new_grid_search_param in enumerate(self.grid_search_params): 
                        found = False
                        for idx_precomputed_params, precomputed_param in enumerate(self.precomputed_result['params']):
                            #to do: verify this with unit tests
                            ### to do >>>>>>>>>>>>>>>> this does not work
                            if self.__compare_two_params(new_grid_search_param, precomputed_param):
#                            if  deep_eq(new_grid_search_param, precomputed_param):

                                matching_pre_computed_idx.append(idx_precomputed_params)
                                matching_pre_computed_params.append(precomputed_param)
                                
                                found = True
                                break
                        if not found:                            
                            reduced_grid_search_params.append(new_grid_search_param)


                    for key, df in self.precomputed_result['processed_results_dict'].items():
                        matching_pre_computed_processed_results_dict[key] = df.iloc[matching_pre_computed_idx]

                    matching_precomputed_result['params'] = matching_pre_computed_params
                    matching_precomputed_result['processed_results_dict'] = matching_pre_computed_processed_results_dict
                    
                    return matching_precomputed_result, reduced_grid_search_params
                
            if self.precomputed_result_found == False:
                reduced_grid_search_params = self.grid_search_params
                matching_precomputed_result = None
                return matching_precomputed_result, reduced_grid_search_params
                 
        else:
             return matching_precomputed_result, self.grid_search_params



            ################# continue from here######################
#            self.grid_search_params = self.grid_search_params.
        
#    def update_precomputed_results(self, X, y, new_params, )    :
        
        
#    def merge_precomputed_results():
        
    



    def process_results(self, cv_results, pipeline):        
        processed_results_dict = self.__get_params_and_scores(cv_results)
        processed_results_dict =  self.__convert_functions_to_class_names(processed_results_dict)        
        #remove features for which multiple variations were not needed
#        processed_results_dict = self.__remove_variance(processed_results_dict, compact_version)        
        processed_results_dict = self.__order_wrt_pipeline(processed_results_dict, pipeline)       
        return processed_results_dict
    
    

    def fit(self, X, y):       



            
        #to do: remove this
        if self.use_precomputed_results:
            matching_precomputed_result, reduced_grid_search_params = self.load_precomputed_results(X,y)
            
#            X = X.to_numpy()
#            y = y.to_numpy()
#            
            if reduced_grid_search_params == []:
                #all parameters found in archive
                self.processed_results_dict = matching_precomputed_result['processed_results_dict']


            else:
                reduced_grid_search_params_copy = copy.deepcopy(reduced_grid_search_params)
                
#                p1 = reduced_grid_search_params[0]['multiregressor__estimator']
#                p1 = str(p1)
                
                self.grid_search_estimator = GridSearchCV(self.pipeline, reduced_grid_search_params_copy, **self.grid_search_arguments)    
                
                t1 = time.time()
                    

#                self.grid_search_estimator.fit(X.to_numpy(), y.to_numpy())                    
                self.grid_search_estimator.fit(X, y)                    

                    
                t2 = time.time()         
                self.processing_time_ = t2-t1   
                
                self.cv_results = self.grid_search_estimator.cv_results_    
                processed_results_dict = self.process_results(self.cv_results, self.pipeline)
                
                self.update_precomputed_result_archive(X, y, reduced_grid_search_params, processed_results_dict)        
                
                p2 = reduced_grid_search_params[0]['multiregressor__estimator']
                p2 = str(p2)                
                
                p3 = reduced_grid_search_params_copy[0]['multiregressor__estimator']
                p3 = str(p3)                     
                
                
                if matching_precomputed_result is not None:
                    #to do: merge here
                    self.processed_results_dict = matching_precomputed_result['processed_results_dict']
    
    
                    #load back precomputed data
                    for key, df in self.processed_results_dict.items():
                        df = df.append(matching_precomputed_result['processed_results_dict'][key], ignore_index=True)     




        else:
            
            
            import dask.array as da
        
#            X_dask = da.from_array(X.to_numpy(), chunks = 500)        
#            y_dask = da.from_array(y.to_numpy(), chunks = 500)
 
#            X_dask = pd.from_pandas(X, npartitions = 2)        
#            y_dask = pd.from_pandas(y, npartitions = 2)
            
            self.grid_search_estimator = GridSearchCV(self.pipeline, self.grid_search_params, **self.grid_search_arguments)        

            t1 = time.time()
            self.grid_search_estimator.fit(X.to_numpy(), y.to_numpy())
#            self.grid_search_estimator.fit(X, y)
#            self.grid_search_estimator.fit(X_dask, y_dask)

            
            t2 = time.time()         
            self.processing_time_ = t2-t1   
            
            self.cv_results = self.grid_search_estimator.cv_results_    
            self.processed_results_dict = self.process_results(self.cv_results, self.pipeline)



        if self.show_plots:
            self.make_plots()



    def update_precomputed_result_archive(self, X, y, new_params, new_processed_results_dict):
        precomputed_result_archive = self.database.load('precomputed_result_archive')



        if self.precomputed_result_found:
#                precomputed_result_archive[self.precomputed_archive_id]['params'] = precomputed_result_archive[self.precomputed_archive_id]['params'].append(new_params)
#                self.precomputed_result['cv_result'].append(new_params)
            
            precomputed_result_archive[self.precomputed_archive_id]['params'] = precomputed_result_archive[self.precomputed_archive_id]['params'] + new_params
            
            for key, df in precomputed_result_archive[self.precomputed_archive_id]['processed_results_dict'].items():
                precomputed_result_archive[self.precomputed_archive_id]['processed_results_dict'][key] = df.append(new_processed_results_dict[key], ignore_index=True)

#                    self.precomputed_result['processed_results_dict'][key] = df.append(new_processed_results_dict[key], ignore_index=True)  
                
            self.database.save(precomputed_result_archive)   
                
        else:            
            if precomputed_result_archive is None:
                precomputed_result_archive = []        

            result = {}            
            result['X'] = X
            result['y'] = y
            result['params'] = new_params
            result['processed_results_dict'] = new_processed_results_dict
            
#            result['cv_result'].append(new_params)
            
            precomputed_result_archive.append(result)            
            self.database.save(precomputed_result_archive)



        
    def make_plots(self):
            self.processed_results_dict = self.__remove_variance(self.processed_results_dict, self.compact_plots)   
            self.processed_results_dict = self.__order_wrt_pipeline(self.processed_results_dict, pipeline)       


            self.scoring = self.grid_search_arguments['scoring']
            
            #make parallel categories plot
            self.plotter.parallel_categories(self.processed_results_dict['parameters_to_plot'] , self.processed_results_dict['scores'], score_name_to_plot = 'mean_test_score', message = self.message)                                
            
            #make box plots for all multiwrappers
            self.parallel.compute(self.processed_results_dict['functions_to_plot'], function = self.make_box_plots)  
            
            #plot all experiment parameters
            experiment_wise_data_df = self.get_experiment_wise_data(self.processed_results_dict)
            self.parallel.compute(experiment_wise_data_df, self.plot_all_experiments)
    
    

    def __dp(self, curr_item = None, prefix = '', depth = 0):
        if type(curr_item) == dict:
            params = []
            for key, value in curr_item.items():            
                if type(key) == str:
                    if prefix == '':
                        res = self.__dp(curr_item = value, prefix = key, depth = depth+1)
                        
                    else:
                        res = self.__dp(curr_item = value, prefix = prefix + '__'+ key, depth = depth+1)
                else:
                    #key is assumed to be a function
                    estimator_key_value = {prefix: [key]}
                    res = self.__dp(curr_item = value, prefix = prefix , depth = depth+1)
                    
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
    #                res = self.__dp(curr_item = value[1], prefix = prefix + '__'+ value[0], depth = depth+1)
    #            else:                
                res = self.__dp(value, prefix , depth = depth+1)
                if type(res) == list:
                    for item in res:
                        params.append(item)
                else:
                    params.append(res)
            return params
            
        else:
            return {prefix: [curr_item]}
    
    
    
       
    def __get_params_and_scores(self, results):        
        parameters_to_plot = pd.DataFrame(results['params'])
        scores = pd.DataFrame()
    
        for key, value in results.items():
            if type(results[key] ) == np.ndarray:
                scores[key] = value
    
        processed_results_dict = {}
        processed_results_dict['parameters_to_plot'] = parameters_to_plot
        processed_results_dict['scores'] = scores        
        return processed_results_dict
    
    
    def __convert_functions_to_class_names(self, processed_results_dict):
        df_local = processed_results_dict['parameters_to_plot'].copy()
        total_rows = df_local.shape[0]
        function_locations_mask = pd.DataFrame(False, index=range(total_rows), columns=df_local.columns)
    
        for column_idx, column in enumerate(df_local.columns):
            for row_idx, value in enumerate(df_local[column]) :
                cell_value = df_local.iloc[row_idx, column_idx]
                if not (isinstance(cell_value, str) or isinstance(cell_value, numbers.Number) ):
                    try:
                        class_name = type(cell_value).__name__
                        
                        if class_name == 'NamedFunctionTransformer':
                            df_local.iloc[row_idx, column_idx] = cell_value._name
                            function_locations_mask.iloc[row_idx, column_idx] = True                        
                            
                        
                        else:                            
                            df_local.iloc[row_idx, column_idx] = type(cell_value).__name__
                            function_locations_mask.iloc[row_idx, column_idx] = True                        
                    except:
                        df_local.iloc[row_idx, column_idx] = str(cell_value)
                        
        processed_results_dict['parameters_to_plot'] = df_local
        processed_results_dict['functions_to_plot'] = df_local.where(function_locations_mask, np.nan)                       
        return processed_results_dict
    
    
    def __remove_variance(self, processed_results_dict, compact_version = False):
        temp_df = processed_results_dict['parameters_to_plot'].copy()        
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
    
    
    def __get_indexes_with_top_function_name(self, all_column_names, top_function_name):
        df = all_column_names.str.split('__', expand = True)
        indexes = df.index[df[0] == top_function_name].tolist()  
        selected_column_names = all_column_names.iloc[indexes]
        return list(selected_column_names)
    
    def __order_wrt_pipeline(self, processed_results_dict, pipeline):
        #extract columns that have a function. As only these column names are found in pipeline 
        #and need to be arranged
        all_column_names = pd.Series(processed_results_dict['functions_to_plot'].columns)
    
        temp_df = processed_results_dict['functions_to_plot'].copy()
        temp_df = temp_df.dropna(axis = 1)    
    
        functions_in_pipeline_ordered = pd.DataFrame(pipeline.steps)
        functions_in_pipeline_ordered = functions_in_pipeline_ordered[0].tolist()
    
        new_order_of_function_columns = []    
    
        for column in functions_in_pipeline_ordered:
            selected_column_names = self.__get_indexes_with_top_function_name(all_column_names, column)
            new_order_of_function_columns.append(selected_column_names)
            
        new_order_of_function_columns = list(itertools.chain.from_iterable(new_order_of_function_columns))
        assert (len (new_order_of_function_columns) == len(all_column_names))
        
        names_of_df_to_reindex = ['functions_to_plot', 'parameters_to_plot']    
        for df_name in names_of_df_to_reindex:
            df = processed_results_dict[df_name]        
            df = df.reindex(columns=new_order_of_function_columns)
            processed_results_dict[df_name] = df
    
        return processed_results_dict
    

    
    
    
    

    
    
    
    
    
    def split_at_char_int_transistion(self, text_str):                       
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
            split_list = re.findall('(\d+|\D+)', element)
            new_list.append(split_list)
        
        new_list = mpu.datastructures.flatten(new_list)         
        return new_list
    
    
    
    def __get_split_scores_df(self, df):
        df_local = df.copy()                    
        index_to_drop = []
    
        for idx, value in enumerate(df.index):
            score_name = self.split_at_char_int_transistion(value)[0]            
            if not score_name == 'split':
                index_to_drop.append(value)                        
        
        df_local = df_local.drop(index_to_drop)
        
        return df_local
                    
                    
    def make_box_plots(self, functions_to_plot,  message = '', x_label = '', y_label = ''):
        for column in functions_to_plot.columns:            
            group_obj =  functions_to_plot.groupby(column)     
            group_dict = group_obj.groups
            group_count_in_column = group_obj.count().shape[0]        

            if group_count_in_column > 0:                
                print('plotting: ', column)
                df_to_plot = pd.DataFrame()

                for key, value in group_dict.items():
                    selected_scores = self.processed_results_dict['scores'].loc[value, ['mean_test_score']]
                    if self.lower_is_better:
                        best_score_row = selected_scores.idxmin()
                    else:
                        best_score_row = selected_scores.idxmax()
                                      
                    extracted_df = self.processed_results_dict['scores'].loc[best_score_row].copy()                    
                    extracted_df = extracted_df.T
                    extracted_df.columns = [key]
                    df_to_plot = df_to_plot.join(extracted_df, how = 'outer')

                split_scores_df = self.__get_split_scores_df(df_to_plot)
                x_label = column.split('__')[-1]
                self.plotter.box_plot_with_mean(split_scores_df, message = column , x_label = x_label, y_label = self.scoring)



    def finder(self, df, row):
        for col in df:
            df = df.loc[(df[col] == row[col]) | (df[col].isnull() & pd.isnull(row[col]))]
        return df.index

    def get_experiment_wise_data(self, processed_results_dict):
        functions_to_plot = processed_results_dict['functions_to_plot'] 
        parameters_to_plot = processed_results_dict['parameters_to_plot']

        functions_to_plot_mask = functions_to_plot.isnull()
        parameters_to_plot_mask = parameters_to_plot.notnull()      
        parameters_only_mask = functions_to_plot_mask & parameters_to_plot_mask        
        unique_parameter_combinations = parameters_only_mask.drop_duplicates()        

        all_df_to_plot = []
        
        for index, row in unique_parameter_combinations.iterrows():
            matching_indexes = self.finder(parameters_only_mask, row)            
            df_to_plot = parameters_to_plot.loc[matching_indexes]
            df_to_plot = df_to_plot.dropna(axis = 1)            
            matching_columns = df_to_plot.columns            
            dict_for_experiment = {}            
            dict_for_experiment['matching_rows'] = matching_indexes
            dict_for_experiment['matching_columns'] = matching_columns
            all_df_to_plot.append(dict_for_experiment)
        return pd.DataFrame(all_df_to_plot).T
    
    

    def __plot_experiment(self, df):
        title = ''
        columns_with_no_varience = []    
        columns_with_numeric_values = []
        columns_with_no_numeric_values = []
        
        for column in df.columns:
            group_obj =  df.groupby(column, observed = True)            
            group_count_in_column = group_obj.count().shape[0]
            group_dict = group_obj.groups

            if group_count_in_column < 2:
                columns_with_no_varience.append(column)   
                title_to_append = list(group_dict)
                title_to_append = title_to_append[0]
                
                if title == '':
                    title = column + ' - ' + title_to_append                    
                else:    
                    title = title + ' - ' + str(title_to_append)
        
            elif is_numeric_dtype(df[column]):
                columns_with_numeric_values.append(column)
                
            else:
                columns_with_no_numeric_values.append(column)              
        df = df.drop(columns = columns_with_no_varience)
    
        metric_to_plot = 'mean_test_score'
        score_df = self.processed_results_dict['scores']
        score_df = score_df.loc[df.index]
        score_column = score_df[metric_to_plot]

        if len(columns_with_numeric_values) == 0:
            return False        
#        plotter.parallel_categories(df, processed_results_dict['scores'] , metric_to_plot, message = title)
        df[metric_to_plot] = score_column

        if len(columns_with_numeric_values) == 2:
            color_scales = self.plotter.color_scales
            color_scales = cl.scales['9']['seq']
            color_scale_iterator = iter(color_scales)
            
            colors = self.plotter.color_list
            colors_iterator = iter(colors)
            
            traces = []

            for non_numeric_column in columns_with_no_numeric_values:

                if len(columns_with_no_numeric_values) > 0:
                    unique_groups_in_non_numeric_column = list(df.groupby(columns_with_no_numeric_values[0]).groups.keys())
                    
                    for group in unique_groups_in_non_numeric_column:
    
                        temp_df = df.where(df[non_numeric_column] == group).copy()
                        temp_df = temp_df.dropna()
    #                    temp_df = temp_df.reset_index()
    
                        x = temp_df[columns_with_numeric_values[0]] 
                        y = temp_df[columns_with_numeric_values[1]]
                        z = temp_df[metric_to_plot]     
                
                    
                        trace = go.Mesh3d(x=x,y=y,z=z,
    #                               alphahull=3,
                                   opacity=.5,
    #                               colorscale=next(color_scale_iterator),
                                   intensity=z,   
    #                               facecolor = z,
                                   name = group,
                                   flatshading = True,
                                   contour_show = True,
                                   contour_width = 10
                                   )
    
                        traces.append(trace)
    
    
    
                        fig = go.Figure(data=trace)
                        fig.update_layout(
                                            title = title + '__' + group,
                                        )
                        fig.update_layout(scene = dict(
                                            xaxis_title= x.name.split('__')[-1],
                                            yaxis_title= y.name.split('__')[-1],
                                            zaxis_title= metric_to_plot),          
                                            )
                        
                        fig.update_layout(
                                           hoverlabel_bgcolor = 'black',
                                        )
            
                        filename = os.path.join(os.getcwd(), title + '__' + group + '.html')   
                        plotly.offline.plot(fig, show_link = True, filename = filename)   
                    
                    
            temp_df = df.dropna()                    
            x = temp_df[columns_with_numeric_values[0]] 
            y = temp_df[columns_with_numeric_values[1]]
            z = temp_df[metric_to_plot]   
        
            if len(columns_with_no_numeric_values) == 0:
#                temp_df = df.dropna()
#                x = temp_df[columns_with_numeric_values[0]] 
#                y = temp_df[columns_with_numeric_values[1]]
#                z = temp_df[metric_to_plot]     
        
            
                trace = go.Mesh3d(x=x,y=y,z=z,
#                               alphahull=3,
                           opacity=.5,
#                               colorscale=next(color_scale_iterator),
                           intensity=z,   
#                               facecolor = z,
                           flatshading = True,
                           contour_show = True,
                           contour_width = 10
                           )
                fig = go.Figure(data=trace)
                
            else:                
                fig = go.Figure(data=traces)
                
            fig.update_layout(
                                title = title,
                            )
            fig.update_layout(scene = dict(
                                xaxis_title= x.name.split('__')[-1],
                                yaxis_title= y.name.split('__')[-1],
                                zaxis_title= metric_to_plot),          
                                )
            
            fig.update_layout(
                               hoverlabel_bgcolor = 'black',
                            )

            filename = os.path.join(os.getcwd(), title + '__' + 'all traces' + '.html')   
            plotly.offline.plot(fig, show_link = True, filename = filename) 
            
        else:
            for column_name in columns_with_numeric_values:    
                if len(columns_with_no_numeric_values) == 0:
#                    df = df.sort_values(by = column_name)
#                    fig = px.line(df, x=column_name, y=metric_to_plot, title = title)                
#                    fig.update_xaxes(title = column_name.split('__')[-1])
#                    fig.show()
                    self.plotter.line_plot(df, x = column_name, y = metric_to_plot, message = title )
                else: 
#                    df = df.sort_values(by = column_name)
#                    fig = px.line(df, x=column_name, y=metric_to_plot, color=columns_with_no_numeric_values[0], title = title)                
#                    fig.update_xaxes(title = column_name.split('__')[-1])
#                    fig.show()
                    self.plotter.line_plot(df, x = column_name, y = metric_to_plot, color = columns_with_no_numeric_values[0], message = title )






    def plot_all_experiments(self, experiment_wise_data_df):
        parameters_to_plot = self.processed_results_dict['parameters_to_plot']
        
        for column in experiment_wise_data_df:
            row_indexes = experiment_wise_data_df[column]['matching_rows']
            column_indexes = experiment_wise_data_df[column]['matching_columns']
            extracted_experiment_df = parameters_to_plot.loc[row_indexes, column_indexes]
            
            self.__plot_experiment(extracted_experiment_df)
            
            

if __name__ == '__main__':
    from mlflow import log_metric, log_param, log_artifacts

 
    from MultiWrappers import MultiTf, MultiRegressor, MultiRegressorWithTargetTransformation, MultiTransformedTargetRegressor, NamedFunctionTransformer
    from SaveAndLoad import SaveAndLoad
    from xgboost.sklearn import XGBRegressor
#    from dask_ml.xgboost import XGBRegressor
#    import xgboost as xgb
#
#    xgb.__version__


    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.datasets import fetch_california_housing
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import LinearRegression, Lasso
    import numpy as np
    from sklearn.feature_selection import SelectFromModel
    from sklearn.svm import LinearSVR
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import f_regression
    from sklearn.feature_selection import RFE
    from sklearn.svm import SVR
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.decomposition import PCA
    
    
#    from MultiWrappers import MultiTransformedTargetRegressor
    from sklearn.preprocessing import PowerTransformer
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression


    global_random_seed = 0
    np.random.seed(global_random_seed)

    #reimport because if an issue where pickle cannot find a class not part of __main__
    import time



    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import cufflinks as cf
    cf.go_offline()
    
    
    import plotly.io as pio
    pio.renderers
    pio.renderers.default = "browser"
    
    import plotly.express as px
    import plotly
    import plotly.graph_objects as go
    

    def get_equally_spaced_numbers_in_range(min_num = 1, max_num =100, total_numbers =10):
        equally_spaced_numbers_list = np.linspace(min_num, max_num,total_numbers)
        equally_spaced_numbers_list = equally_spaced_numbers_list.astype(int)
        equally_spaced_numbers_list = np.unique(equally_spaced_numbers_list)
        equally_spaced_numbers_list = equally_spaced_numbers_list.tolist()
        return equally_spaced_numbers_list
    
    
    database = SaveAndLoad('/media/pt/hdd/Auto EDA Results/regression/results/pickles')   
    
    combined_categorical_df = database.load('combined_categorical_df')
    combined_continuous_df = database.load('combined_continuous_df')
    y_train = database.load('y_train')
    
    joint_df = database.load('joint_df')
    
    
#    memory = '/media/pt/hdd/Auto EDA Results/regression/results/memory'
    memory = '/media/pt/hdd/from NVME/sklearn_memory/memory'



    ExtraTreesRegressor_params =    {
                                        'max_depth': get_equally_spaced_numbers_in_range(1,100,10) , 
                                        'n_estimators': get_equally_spaced_numbers_in_range(1,2000,10),

#                                        'max_depth': get_equally_spaced_numbers_in_range(1,4,4) ,
#                                        'n_estimators': get_equally_spaced_numbers_in_range(2,4,4),
                                        'random_state' : [global_random_seed]
#                                        'max_depth': [1] , 
#                                        'n_estimators': [1], 
                                        
                                    }
    
    
    KNeighborsRegressor_params =    {
                                        'n_neighbors' : [2,3,4],
#                                        'n_neighbors' : [2],
                                    }
    
    
    bayesianRidge_params =          {
                                        'n_iter' : get_equally_spaced_numbers_in_range(1,2000,10)
#                                        'n_iter' : [24]
                                    }
    
    
    estimator_list =                [   
                                        {BayesianRidge() : bayesianRidge_params},
                                        DecisionTreeRegressor(),
                                        {KNeighborsRegressor() : KNeighborsRegressor_params},
                                        {ExtraTreesRegressor() : ExtraTreesRegressor_params}
                                    ]
    
    
    iterative_imputer_params =      {
                                        'estimator' : estimator_list,
                                        'missing_values' : [np.nan],
                                        'random_state' : [global_random_seed]
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
#                                        'transformer' : [iterative_imputer_dict]
                                        'transformer' : [iterative_imputer_dict, simple_imputer_dict]
                                        
                                    }
    
    
    power_transformer_params =      {
                                        'method' : ['yeo-johnson', 'box-cox'],
                                        'standardize' : [False],
                                        'copy' : [False]
                                    }
    
    
    power_transformer_dict =        {
                                        PowerTransformer() : power_transformer_params
                                    }
    
      
    target_transformer_log_exp = NamedFunctionTransformer(func = np.log, inverse_func = np.exp, name = 'log_exp')   
    target_transformer_log1p_expm1 = NamedFunctionTransformer(func = np.log1p, inverse_func = np.expm1, name = 'log1p_expm1')   
                                

    xgboost_dict = {    
                        XGBRegressor(): [ {'n_jobs' : [-1] } ] 
                    }

    MultiRegressorWithTargetTransformation_params = {   
#                                                        'regressor' : [xgboost_dict],

                                                        'regressor' : [xgboost_dict, AdaBoostRegressor(), KNeighborsRegressor() ],
#                                                        'regressor' : [AdaBoostRegressor(), KNeighborsRegressor() ],
    
                #                                        'func_inverse_func_pair' : [(np.log, np.exp), (np.log1p, np.expm1)],
                #                                        'inverse_func' : [np.exp]
                                                        'transformer' : [target_transformer_log1p_expm1]                                                        
                                                    }
   
#    MultiTransformedTargetRegressor_params =        {   
#                                        'transformer' : [PowerTransformer()]
#                                    }
    
#    multi_regressor_params =        {   
#                                        'estimator' : [XGBRegressor() ]
#                                    }    
    
#    multi_regressor_params =      {   
#                                        'estimator' : [XGBRegressor(), AdaBoostRegressor(), KNeighborsRegressor() ]
#                                  }

    
    
    selectfrommodel_params =        {
                                        'estimator' : [Lasso(alpha=0.1)]
                                    }
    
        
    selectfrommodel_dict =          {
#                                        SelectFromModel(LinearSVR()) : SelectFromModel_params
                                        SelectFromModel(Lasso(alpha=0.1))  

                                    }
    
    multi_selector_params =        {    
#                                        'transformer' : [selectfrommodel_dict]  ,
#                                         'transformer' : [SelectFromModel(LinearSVR()) ]
                                    }        
    

    dasktf_params = {
                        'estimator' : [simple_imputer_dict, iterative_imputer_dict]
                    }
 
    dask_MultiRegressorWithTargetTransformation_dict = {
                                                            MultiRegressorWithTargetTransformation(): [MultiRegressorWithTargetTransformation_params]
                                                        }
    
    
    DaskMultiRegressorWithTargetTransformation_params     = {
                                                                'estimator': dask_MultiRegressorWithTargetTransformation_dict
                                                            }
    
    TransformedTargetRegressor_params = {
#                                           'regressor' : [KNeighborsRegressor() ],

                                           'regressor' : [XGBRegressor(), AdaBoostRegressor(), KNeighborsRegressor() ],
                                           'func': [np.log1p], 
                                           'inverse_func': [np.expm1]                                            
                                       }
   
    TransformedTargetRegressor_dict = {
                                            TransformedTargetRegressor() : TransformedTargetRegressor_params
                                        }
    
    DaskMultiRegressorWithTargetTransformation_params_2     = {
                                                                'estimator': TransformedTargetRegressor_dict
                                                            }
    
    
#    config_dict_1 =                   {   
#                                        'multitf' : multitf_params,
#                                        'multiregressor' : multi_regressor_params                                        
#                                    }    
#    
#    
#    config_dict_2 =                   {   
#                                        'multitf' : multitf_params,
##                                        'selectfrommodel' :selectfrommodel_params,
#                                        'multiregressor' : multi_regressor_params                                        
#                                    }  
#
    config_dict_3 =                   {   
                                    'multitf' : multitf_params,
                                    'MultiRegressorWithTargetTransformation' : MultiRegressorWithTargetTransformation_params                                        
                                } 
        
    config_dict_4 =                   {   
                                    'dasktf' : dasktf_params,
                                    'MultiRegressorWithTargetTransformation' : MultiRegressorWithTargetTransformation_params                                        
                                } 

    config_dict_5 =                   {   
                                    'dasktf' : dasktf_params,
                                    'DaskMultiRegressorWithTargetTransformation' : DaskMultiRegressorWithTargetTransformation_params_2                                        
                                } 

    config_dict_6 =                   {   
                                    'multitf' : multitf_params,
                                    'DaskMultiRegressorWithTargetTransformation' : DaskMultiRegressorWithTargetTransformation_params_2                                        
                                }     


    config_dict = config_dict_5
    
    
#    steps =     [
#                    ('multitf' , MultiTf() ), 
#                    ('scaler' , StandardScaler() ),
#                    ('pca', PCA(n_components = .999)),
##                    ('selectfrommodel', SelectFromModel(Lasso(alpha=0.1))),
#                    ('multiregressor' , MultiRegressor() ) 
#                ]
    
    
    steps_3 = [
                ('multitf' , MultiTf() ), 
                ('scaler' , StandardScaler() ),
                ('pca', PCA(n_components = .999)),
                ('MultiRegressorWithTargetTransformation' , MultiRegressorWithTargetTransformation() ) 
            ]
    
    
    steps_4 = [
            ('dasktf' , ParallelPostFit() ), 
            ('scaler' , StandardScaler() ),
            ('pca', PCA(n_components = .999)),
            ('MultiRegressorWithTargetTransformation' , MultiRegressorWithTargetTransformation() ) 
        ]

    steps_5 = [
            ('dasktf' , ParallelPostFit() ), 
            ('scaler' , StandardScaler() ),
            ('pca', PCA(n_components = .999)),
            ('DaskMultiRegressorWithTargetTransformation' , ParallelPostFit() ) 
        ]
 
    steps_6 = [
            ('multitf' , MultiTf() ), 
            ('scaler' , StandardScaler() ),
            ('pca', PCA(n_components = .999)),
            ('DaskMultiRegressorWithTargetTransformation' , ParallelPostFit() ) 
        ]    
    steps = steps_5
        
    #    memory = '/media/pt/hdd/Auto EDA Results/regression/results/memory'
    pipeline = Pipeline( memory = memory, steps = steps)    
    

#    from dask.distributed import Client
#    client = Client(processes=True)

    database = SaveAndLoad('/media/pt/hdd/Auto EDA Results/unit_tests/pickles')
    plot_dir = ('/media/pt/hdd/Auto EDA Results/unit_tests/analyzer_plots')
    

    
    from distributed import Client
    from distributed.deploy.local import LocalCluster
    memory = '/media/pt/nvme/dask_memory'
    
    cluster = LocalCluster(n_workers = 4, threads_per_worker = 6, memory_limit='128GB', local_dir=os.path.join(memory)  )
    client = Client(cluster)
    print(client)
    
    
    
    
#    client = Client(memory_limit='240GB')
#    print(client)

#    from distributed import Client
#    client = Client()
#    client



    auto_imputer = Analyzer(plot_dir = plot_dir, message = 'Imputer Analysis', database = database, debug_mode = True, show_plots = True, use_precomputed_results = False, use_dask = True, prefix_for_parameters = '')
    auto_imputer.gridsearchcv(pipeline, config_dict, cv = 10, scoring = 'neg_mean_squared_log_error', scheduler=client, refit = False, cache_cv = True)
#    auto_imputer.gridsearchcv(pipeline, config_dict, cv = 10, n_jobs = -1, scoring = 'neg_mean_squared_log_error', refit = False)

    p = auto_imputer.grid_search_params

  
    
    auto_imputer.fit(joint_df, y_train)  
    





    
#    database = SaveAndLoad('/media/pt/hdd/Auto EDA Results/unit_tests/pickles')
#    plot_dir = ('/media/pt/hdd/Auto EDA Results/unit_tests/analyzer_plots')
#
#    auto_imputer = Analyzer(plot_dir = plot_dir, message = 'Imputer Analysis', database = database, debug_mode = True, show_plots = True, use_precomputed_results = False, use_dask = True, prefix_for_parameters = '')
#    auto_imputer.gridsearchcv(pipeline, config_dict, cv = 10, n_jobs = -1, scoring = 'neg_mean_squared_log_error', pre_dispatch='2*n_jobs', verbose = 1)
#
#    transformer = PowerTransformer(method='box-cox', standardize=False, copy=True)
#    y_train_np = y_train.values
#    y_train_np = y_train_np.reshape(-1,1)
#    transformer.fit(y_train_np)
#    y_train_transformed = transformer.transform(y_train.values.reshape(-1,1))
#  
#    t = pd.DataFrame(y_train_transformed)
#    t.hist()
#    
#    auto_imputer.fit(joint_df, y_train_transformed)    
#
#












#
#
#    config_dict =                   {   
#                                        'multitf' : multitf_params,
##                                        'multiselector' : multi_selector_params, 
#                                        'MultiTransformedTargetRegressor' : MultiTransformedTargetRegressor_params                                        
#                                    }    
#    steps = [
#                ('multitf' , MultiTf() ), 
#                ('scaler' , StandardScaler() ),
#                ('pca', PCA(n_components = .999)),
##                ('multiselector', RFE(SVR(kernel="linear"), step=.1)) ,                 
#                ('MultiTransformedTargetRegressor' , MultiTransformedTargetRegressor() ) 
#            ]
#        
#    pipeline = Pipeline( memory = memory, steps = steps)    
#    
#    
#
#    from MultiWrappers import MultiTransformedTargetRegressor
#    from sklearn.preprocessing import PowerTransformer
#    from sklearn.preprocessing import QuantileTransformer
#    from sklearn.compose import TransformedTargetRegressor
#    
#    
#    database = SaveAndLoad('/media/pt/hdd/Auto EDA Results/unit_tests/pickles')
#    plot_dir = ('/media/pt/hdd/Auto EDA Results/unit_tests/analyzer_plots')
#
#    auto_imputer = Analyzer(plot_dir = plot_dir, message = 'Imputer Analysis', database = database, debug_mode = True, show_plots = True, use_precomputed_results = False, use_dask = True, prefix_for_parameters = 'regressor')
#    auto_imputer.gridsearchcv(pipeline, config_dict, cv = 10, n_jobs = -1, scoring = 'neg_mean_squared_log_error', pre_dispatch='2*n_jobs', verbose = 1)
#    
#    params = auto_imputer.grid_search_params    
##    transformer_parameters = {'transformer': [PowerTransformer]}    
##    params.append(transformer_parameters)
#
#    transformer = PowerTransformer(method='yeo-johnson', standardize=False, copy=True)
#
#    target_regressor = TransformedTargetRegressor(regressor=pipeline, transformer = transformer)
#    
#    target_regressor_cv = GridSearchCV(target_regressor, params, cv=5)
#    target_regressor_cv.fit(joint_df, y_train)
#    
#    
#    



        
#
#
#            x = df[columns_with_numeric_values[0]]
#            y = df[columns_with_numeric_values[0]]
#            z = score_column
#            
#            color = df[columns_with_no_numeric_values[0]]
#                
#            trace_train = go.Mesh3d(x=x,y=y,z=z,
#                       alphahull=3,
#                       opacity=.5,
#                       colorscale="Reds",
#                       intensity=color,   
#                       
#                       )
#            
#    #        trace_test = go.Mesh3d(x=x,y=y,z=z,
#    #                   alphahull=3,
#    #                   opacity=.5,
#    #                   colorscale="Greens",
#    #                   intensity=color,                        
#    #                   )
#            
#            traces = [trace_train]
#                
#            plotly.offline.plot(traces)

            

#    
#        fig = px.line(df, x = coluns_with_numeric_values, y=metric_to_plot, color='country')
#        fig.show()
    
            
        


    

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







