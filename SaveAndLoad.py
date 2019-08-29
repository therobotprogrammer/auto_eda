#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 11:13:28 2019

@author: pt
"""
#
#import inspect
#import pickle
#import os
#
#class SaveAndRetrieve:
#    def __init__(self, save_directory):
#        self.save_directory = save_directory
#        
#        
#    def autodict(self, *args):
#        get_rid_of = ['autodict(', ',', ')', '\n']
#        calling_code = inspect.getouterframes(inspect.currentframe())[1][4][0]
#        calling_code = calling_code[calling_code.index('autodict'):]
#        for garbage in get_rid_of:
#            calling_code = calling_code.replace(garbage, '')
#        var_names, var_values = calling_code.split(), args
#        dyn_dict = {var_name: var_value for var_name, var_value in
#                    zip(var_names, var_values)}
#        return dyn_dict
#    
#    def reterive_variable_name(self, variable):
#        dict_of_names = self.autodict(variable)
#        variable_name = list(dict_of_names.keys())[0]        
#        return variable_name
#    
#    def save(self, variable):
#        if not os.path.isdir(self.save_directory ):
#           os.makedirs(self.save_directory )
#
#        variable_name = self.reterive_variable_name(variable)
#        pickle_file_name = os.path.join(self.save_directory , str(self.reterive_variable_name(variable) ) + '.pickle')
#        
#        print('Saving: ', pickle_file_name)
#        pickle_out = open(pickle_file_name,"wb")
#        pickle.dump(variable, pickle_out)
#        pickle_out.close()
#
#
#if __name__ == '__main__':
#    SaveAndRetrieve_obj = SaveAndRetrieve('/media/pt/hdd/Auto EDA Results/regression/results/pickles')
#    SaveAndRetrieve_obj.save(combined_categorical_df)
#
#





import inspect
import pickle
import os


class SaveAndLoad:
    def __init__(self, save_directory):   
        if not os.path.isdir(save_directory ):
            os.makedirs(save_directory )
        self.save_directory = save_directory

    def save(self, *args, variable_name = ''):
        if variable_name == '':
            get_rid_of = ['save(', ',', ')', '\n']
            calling_code = inspect.getouterframes(inspect.currentframe())[1][4][0]
            calling_code = calling_code[calling_code.index('save'):]
            
            for garbage in get_rid_of:
                calling_code = calling_code.replace(garbage, '')
            var_names, var_values = calling_code.split(), args
            dyn_dict = {var_name: var_value for var_name, var_value in
                        zip(var_names, var_values)}
            variable_name = list(dyn_dict.keys())[0]       
            variable_name = str(variable_name)
            
        variable_value = args        
        self.__helper_save_with_pickle(variable_name, variable_value)


    def __helper_save_with_pickle(self, variable_name, variable_value):
        pickle_file_name = os.path.join(self.save_directory , str(variable_name) + '.object')
        print('Saving: ', pickle_file_name)
        pickle_out = open(pickle_file_name,"wb")
        pickle.dump(variable_value, pickle_out)
        pickle_out.close()        

        
    def load(self, variable_name):
        return(self.__helper_load_with_pickle(variable_name))


    def __helper_load_with_pickle(self, variable_name):        
        filename = os.path.join(self.save_directory, variable_name  + '.object')
        
        if os.path.isfile(filename):
            print('Loading:' , filename)
            file = open(filename, 'rb')
            data = pickle.load(file)[0]
            file.close()
        else:
            data = None
        
        return data
    


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    import pandas as pd
    
    
    database = SaveAndLoad('/media/pt/hdd/Auto EDA Results/unit_tests/pickles')   

    iris_data = load_iris()    
    iris_data_df_before_saving = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)    
    database.save(iris_data_df_before_saving)    
    iris_data_df_before_saving_after_saving = database.load('iris_data_df_before_saving') 
    assert iris_data_df_before_saving_after_saving.equals(iris_data_df_before_saving)

    example_dict_before_saving = {1:"6",2:"2",3:"f"}
    database.save(example_dict_before_saving)   
    example_dict_after_saving = database.load('example_dict_before_saving')    
    assert(example_dict_before_saving == example_dict_after_saving)    
    
    print('Tests Passed!!!')
    
    
    























