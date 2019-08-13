#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:39:15 2019

@author: pt
"""

ExtraTreesRegressor_params = {
                                    'max_depth': [1, 2, 3], 
                                    'n_estimators': [1,10,100,1000]
                             }



estimator_list = [   
                    'BayesianRidge()', 
                    'DecisionTreeRegressor()',                    
                    'KNeighborsRegressor()',
                    {'ExtraTreesRegressor()' : ExtraTreesRegressor_params}
                ]


iterative_imputer_params =  {
                                'estimator' : estimator_list,
                                'missing_values' : [np.nan]                                                   
                            }


param_list = []

curr_dict = iterative_imputer_params


import itertools



def dp(curr_item = None, prefix = '', depth = 0):
    if type(curr_item) == dict:
        params = []
        for key, value in curr_item.items():
            local_params = [] 
            print(key, ':', value)            
            res = dp(curr_item = value, prefix = prefix + '__'+ key, depth = depth+1)
            
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
        return {prefix: curr_item}
        
        
temp = dp(curr_dict)

print(temp)






































