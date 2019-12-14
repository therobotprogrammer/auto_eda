#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 21:41:09 2019

@author: pt
"""
import os
from IPython.display import display, HTML

results_dir = '/home/pt/Desktop/FINAL REPORT/Source Files/Exploratory Data Analysis'
graphs = os.listdir(results_dir)
graphs.sort()


if not os.path.isdir(results_dir):
    os.mkdir(results_dir)


def report_block_template(report_type, graph_url, caption=''):
    if report_type == 'interactive':
        graph_block = '<iframe style="border: none;" src="{graph_url}" width="100%" height="600px"></iframe>'
    elif report_type == 'static':
        graph_block = (''
            '<a href="{graph_url}" target="_blank">' # Open the interactive graph when you click on the image
                '<img style="height: 400px;" src="{graph_url}.png">'
            '</a>')

    report_block = ('' +
        graph_block + 
        '{caption}' + # Optional caption to include below the graph
        '<br>'      + # Line break
        '<a href="{graph_url}" style="color: rgb(190,190,190); text-decoration: none; font-weight: 200;" target="_blank">'+ 
            'Click to comment and see the interactive graph' + # Direct readers to Plotly for commenting, interactive graph
        '</a>' + 
        '<br>' + 
        '<hr>') # horizontal line                       
    return report_block.format(graph_url=graph_url, caption=caption)


interactive_report = ''
static_report = ''

for graph_url in graphs:
    _static_block = report_block_template('static', graph_url, caption='')
    _interactive_block = report_block_template('interactive', graph_url, caption='')
    static_report += _static_block
    interactive_report += _interactive_block

a = HTML(interactive_report)
 
html = a.data
with open(results_dir + '/Report - Exploratory Data Analysis.html', 'w') as f:
    f.write(html)    
    
    
    
    
    
    
    
    
    
    