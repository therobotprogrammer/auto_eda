#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:39:15 2019

@author: pt
"""



import itertools
import numpy as np


ExtraTreesRegressor_params = {
                                    'max_depth': [1, 2, 3], 
                                    'n_estimators': [1,10,100,1000]
                             }


KNeighborsRegressor_params = {
                                'n_neighbors' : [4]
                            }


estimator_list = [   
                    BayesianRidge(),
                    DecisionTreeRegressor(),                    
                    {KNeighborsRegressor() : KNeighborsRegressor_params},
                    {ExtraTreesRegressor() : ExtraTreesRegressor_params}
                ]


iterative_imputer_params =  {
                                'estimator' : estimator_list,
                                'missing_values' : [np.nan]                                                   
                            }


iterative_imputer_dict = {
                            IterativeImputer() : iterative_imputer_params
                        }

grid_params = {
                    'transformer' : iterative_imputer_dict
                }


param_list = []

curr_dict = iterative_imputer_params



def dp(curr_item = None, prefix = '', depth = 0):
    if type(curr_item) == dict:
        params = []
        for key, value in curr_item.items():
            local_params = [] 
            print(key, ':', value)   
            
            if type(key) == str:                
                res = dp(curr_item = value, prefix = prefix + '__'+ key, depth = depth+1)
            else:
                #key is assumed to be a function
                estimator_key_value = {prefix: [key]}
                
                print('>>', estimator_key_value)
                res = dp(curr_item = value, prefix = prefix , depth = depth+1)
                
                for sub_dict in res:
                    sub_dict.update(estimator_key_value)
#                res.append(estimator_key_value)
                print('>>>>', res)       
                
#                res.update(estimator_key_value)
            params.append(res)
            
#        r = list(itertools.product(params))

        permutations = []
        
        for element in itertools.product(*params):
            
            d = {}
            
            
            for dict_value in element:
                d.update(dict_value)
            
            permutations.append(d)
#            print(element)
#            ans = res
#            if type(res) == list:
#                for item in res:
#                    local_params.append(item)
#                    
#            else:
#                local_params.append(res)
                
            

#        params = itertools.chain(*params)
#        flat_list = [item for sublist in params for item in sublist]

        return permutations        
                
    elif type(curr_item) == list:
        params = []
        
        for value in curr_item:
            if value == 'ExtraTreesRegressor()':
                print('found')
                
            if type(value) == tuple:
                res = dp(curr_item = value[1], prefix = prefix + '__'+ value[0], depth = depth+1)

            else:                
                res = dp(value, prefix , depth = depth+1)
                
                
            if type(res) == list:
                for item in res:
                    params.append(item)
                    
            else:
                params.append(res)
                
#            params.append(res)
        
#        params = itertools.chain(*params)
            

        return params
        
    else:
#        print(prefix , ':', curr_item)
#        print()
        return {prefix: [curr_item]}
        
        
grid_search_params_dp = dp(grid_params, prefix = 'gowide')

print(grid_search_params_dp)






































