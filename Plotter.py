#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 16:38:25 2019

@author: pt
"""




import pandas as pd
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


class Plotter:
    
    def __init__(self, top_save_directory):   
        if not os.path.isdir(top_save_directory):
            os.makedirs(top_save_directory)
            
        self.top_save_directory = top_save_directory  
        self.current_directory  = top_save_directory
#        self.set_current_dir('')
        
        
    def set_current_dir(self, local_folder_name): 
        if local_folder_name == '':
            current_directory = self.top_save_directory
            
        elif os.path.isdir(local_folder_name):
            current_directory = self.local_folder_name
           
        else:
            current_directory = os.path.join(self.top_save_directory , local_folder_name)
            
        if not os.path.isdir(current_directory):
            os.makedirs(current_directory)        
        
        print(current_directory)
        print()
        self.current_directory = current_directory
        
        
        
        
    def plot_cat_cat(self, x,y):            
        message = 'Plotter cat-cat' + x.name + 'vs target '+ y.name
        print(message)
        
        plt.figure()
        ax = sns.catplot(x= x, hue = y, kind = 'count', height=6)     
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
        plt.tight_layout()
        

        file_name = os.path.join(self.current_directory + '/' + message + '.jpg')
        plt.savefig(file_name, dpi = 1200)
        
        plt.show()
    
        
    def plot_cat_cont(self, x,y, kde = True):
        message = 'Plotter cat-cont  ' + x.name + 'vs target '+ y.name
        print(message)
        plt.figure()
        
        temp_df = pd.DataFrame()        
        temp_df[x.name] = x
        temp_df[y.name] = y
        
        sns.FacetGrid(temp_df, hue = y.name, height=6).map(sns.kdeplot, y.name , vertical = False).add_legend()   

#        sns.FacetGrid(x, hue = y, height=6).map(sns.kdeplot, x).add_legend() 

      
        plt.tight_layout()
        
        file_name = os.path.join(self.current_directory + '/' + message + '.jpg')
        plt.savefig(file_name, dpi = 1200)
        
        plt.show()
        
    
    
    def plot_cont_cont(self, x,y):
        message = 'Plotter cat-cont  ' + x.name + 'vs target '+ y.name
        print(message)
        plt.figure()
#        sns.FacetGrid(train, hue = "Survived", height=6).map(sns.kdeplot, column).add_legend()          
        ax = sns.scatterplot(x = x, y = y, hue = y)
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
        plt.tight_layout()
        
        file_name = os.path.join(self.current_directory + '/' + message + '.jpg')
        plt.savefig(file_name, dpi = 1200)
        
        plt.show()

    def plot_cont_cat(self, x,y):
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
        
        file_name = os.path.join(self.current_directory + '/' + message + '.jpg')
        plt.savefig(file_name, dpi = 1200)
        
        plt.show()
        
    def box_plot_df(self, df_local, message = ''):
        df_local.iplot(kind='box', boxpoints='outliers', title = message + ' - Box Plot')


    def box_plot_plotly_express(self, df_local, message = ''):
        filename = os.path.join(self.current_directory, message + '_box_plot.html')   
        plotly.offline.plot(fig, show_link = True, filename = filename)
        

    def parallel_coordinates(self, input_df, target, message, labels_dict = None):
        combined_local_df = input_df.join( target, how = 'outer')      
    
        if isinstance(target, pd.DataFrame):    
            target_name = target.columns[0]
            
        else:
            #target is a series
            target_name = target.name

        fig = px.parallel_coordinates(combined_local_df, color=target_name, labels=labels_dict,
                                     color_continuous_scale=px.colors.diverging.Tealrose,
                                     color_continuous_midpoint=2 )

        filename = os.path.join(self.current_directory, message + '_parallel_plot.html')   
        plotly.offline.plot(fig, show_link = True, filename = filename)



    def parallel_categories(self, parameters, scores, score_name_to_plot = 'mean_test_score', message = ''): 
        parameters_local = parameters.copy()
        scores_local = scores[score_name_to_plot].copy()        
        temp_df = parameters_local.join(scores_local, how = 'outer')    
        temp_df = temp_df.sort_values('mean_test_score', ascending = False)        
        
        fig = px.parallel_categories(temp_df, color = 'mean_test_score', color_continuous_scale=px.colors.sequential.Inferno)
        filename = os.path.join(self.current_directory, message + '_parallel_categories_plot.html')   
        plotly.offline.plot(fig, show_link = True, filename = filename)        
    

        

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/home/pt/Documents/auto_eda')   
    from SaveAndLoad import SaveAndLoad    
    import plotly.express as px
    import plotly

    iris = px.data.iris()    
    target = iris['species_id']
    iris_df = iris.drop(['species', 'species_id'], axis = 1)   
    
    plot_dir = '/media/pt/hdd/Auto EDA Results/unit_tests/plots'
    plotter = Plotter(plot_dir)    
    plotter.parallel_coordinates(iris_df, target, message = 'iris')

    database = SaveAndLoad('/media/pt/hdd/Auto EDA Results/regression/results/pickles')   
    target = database.load('y_train')  
    combined_continuous_df = database.load('combined_continuous_df')

    plotter = Plotter(plot_dir)    
    plotter.parallel_coordinates(combined_continuous_df, target, message = 'House Price - continuous data')

#    plotter = Plotter(plot_dir)    
#    plotter.parallel_coordinates(parameters_to_plot,  results['mean_test_score'], message = 'House Price - continuous data')
#
#    




