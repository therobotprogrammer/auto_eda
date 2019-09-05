#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 12:11:29 2019

@author: pt
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 20:08:46 2019

@author: pt
"""





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os


from concurrent.futures import ProcessPoolExecutor







from missingpy import KNNImputer, MissForest


import plotly
import plotly.graph_objs as go
import cufflinks as cf
cf.go_offline()


import plotly.io as pio
pio.renderers
pio.renderers.default = "browser"




    
directory = '/media/pt/hdd/Auto EDA Results/regression'   

file_path = os.path.join(directory, 'train.csv')
train = pd.read_csv(file_path, index_col = False)
target_column = 'SalePrice'

    


y_train = train[target_column].copy(deep = True)  

 





y = y_train.values


global_verify_parallel_execution = False
global_debug_mode = True


from funcy import join

#
#def parallise(df_input, function, partitions=None, processes=None):
#    # calculate features in paralell by splitting the dataframe into partitions and using paralell processes
#    if partitions == None:        
#        partitions = global_cores        
#    if processes == None:        
#        processes = global_cores
#       
#    if global_debug_mode == True:
#        compiled_result_serial = function(df_input)
#        return compiled_result_serial
#
#    else:
#            df_split = np.array_split(df_input, partitions, axis=1)  # split dataframe into partitions column wise
#            
#            
#            with ProcessPoolExecutor(processes) as pool:     
#                results_generator = pool.map(function, df_split)             
#                results_as_splits = list(results_generator)
#           
#            compiled_result_parallel = [] 
#            
#            if isinstance(results_as_splits[0], tuple):                
#                
#                items_per_split =  len(results_as_splits[0])                      
#                
#                for index in range(0, items_per_split):
#                    splits_to_concatenate = []
#    
#                    for split in results_as_splits:
#                        
#                        #If multiple values are returned, then they are a tuple. They have to be accessed as index. 
#                        if isinstance(compiled_result_serial, pd.DataFrame) :
#                            splits_to_concatenate.append(split[index].T) 
#                         
#                        else:
#                            splits_to_concatenate.append(split[index])                            
#                   
#                    #concatenate based on datatype
#                    if isinstance(splits_to_concatenate[0], pd.DataFrame):
#                        joined_data_df = pd.concat(splits_to_concatenate, axis = 1) 
#                        compiled_result_parallel.append(joined_data_df)                        
#                   
#                    else:
#                        joined_data = join(splits_to_concatenate)
#                        compiled_result_parallel.append(joined_data)
#                            
#                compiled_result_parallel = tuple(compiled_result_parallel)
# 
#    
#            else:
#                    # If a tuple was not returned, then a was returned by each split
#                    splits_to_concatenate = []
#                    
#                    for split in results_as_splits:
#                            
#                        splits_to_concatenate.append(split.T)                            
#                   
#                    #concatenate based on datatype
#                    if isinstance(splits_to_concatenate[0], pd.DataFrame):
#                        joined_data_df = pd.concat(splits_to_concatenate, axis = 1)      
#
#                        compiled_result_parallel.append(joined_data_df)                
#                    else:
#                        joined_data = join(splits_to_concatenate)
#                        compiled_result_parallel.append(joined_data)
#                        
#                    #since original function returned only one item, 
#                    #this function also returns only 1 item and not a tuple of items
#                    
#                    compiled_result_parallel = compiled_result_parallel[0]      
#                    compiled_result_parallel = compiled_result_parallel.T
#    
#            if global_verify_parallel_execution == True:
#                compiled_result_serial = function(df_input)
#                
#                if isinstance(compiled_result_serial, tuple):
#                    for index, serial_item in enumerate(compiled_result_serial) :
#                        if isinstance(serial_item, pd.DataFrame) :
#                            parallel_item = compiled_result_parallel[index]
#                            assert serial_item.equals(parallel_item), 'Serial and Parallel compution do not match'
#                        
#                        else:
#                            assert serial_item == compiled_result_parallel[index] , 'Serial and Parallel compution do not match'
#                            
#                else:
#                        if isinstance(compiled_result_serial, pd.DataFrame) :
#                            assert compiled_result_serial.equals( compiled_result_parallel ), 'Serial and Parallel compution do not match'
#                        
#                        else:
#                            assert compiled_result_serial == compiled_result_parallel , 'Serial and Parallel compution do not match'                     
#    
#    return compiled_result_parallel
#









#from sklearn import datasets
## Load data and select first column
#data_set = datasets.load_breast_cancer()
#y=data_set.data[:,0]


import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler
import scipy.stats
import matplotlib.pyplot as plt
from scipy.stats._continuous_distns import _distn_names



# Create an index array (x) for data

x = np.arange(len(y))
size = len(y)





plt.hist(y)
plt.show()


y_df = pd.DataFrame(y, columns=['Data'])
y_df.describe()



sc=StandardScaler() 
yy = y.reshape (-1,1)
sc.fit(yy)
y_std =sc.transform(yy)
y_std = y_std.flatten()
y_std
del yy






# Set list of distributions to test
# See https://docs.scipy.org/doc/scipy/reference/stats.html for more

# Turn off code warnings (this is not recommended for routine use)
import warnings
warnings.filterwarnings("ignore")

# Set up list of candidate distributions to use
# See https://docs.scipy.org/doc/scipy/reference/stats.html for more

long_version = [
                    'levy_stable',

                ]
                
                
#                
#dist_names = [  'alpha',
#                'anglit',
#                'arcsine',
#                'argus',
#                'beta',
#                'betaprime',
#                'bradford',
#                'burr',
#                'burr12',
#                'cauchy',
#                'chi',
#                'chi2',
#                'cosine',
#                'crystalball',
#                'dgamma',
#                'dweibull',
#                'erlang',
#                'expon',
#                'exponnorm',
#                'exponweib',
#                'exponpow',
#                'f',
#                'fatiguelife',
#                'fisk',
#                'foldcauchy',
#                'foldnorm',
#                'frechet_r',
#                'frechet_l',
#                'genlogistic',
#                'gennorm',
#                'genpareto',
#                'genexpon',
#                'genextreme',
#                'gausshyper',
#                'gamma',
#                'gengamma',
#                'genhalflogistic',
#                'gilbrat',
#                'gompertz',
#                'gumbel_r',
#                'gumbel_l',
#                'halfcauchy',
#                'halflogistic',
#                'halfnorm',
#                'halfgennorm',
#                'hypsecant',
#                'invgamma',
#                'invgauss',
#                'invweibull',
#                'johnsonsb',
#                'johnsonsu',
#                'kappa4',
#                'kappa3',
#                'ksone',
#                'kstwobign',
#                'laplace',
#                'levy',
#                'levy_l',
#                'logistic',
#                'loggamma',
#                'loglaplace',
#                'lognorm',
#                'lomax',
#                'maxwell',
#                'mielke',
#                'moyal',
#                'nakagami',
#                'ncx2',
#                'ncf',
#                'nct',
#                'norm',
#                'norminvgauss',
#                'pareto',
#                'pearson3',
#                'powerlaw',
#                'powerlognorm',
#                'powernorm',
#                'rdist',
#                'reciprocal',
#                'rayleigh',
#                'rice',
#                'recipinvgauss',
#                'semicircular',
#                'skewnorm',
#                't',
#                'trapz',
#                'triang',
#                'truncexpon',
#                'truncnorm',
#                'tukeylambda',
#                'uniform',
#                'vonmises',
#                'vonmises_line',
#                'wald',
#                'weibull_min',
#                'weibull_max',
#                'wrapcauchy']



dist_names = _distn_names
if 'levy_stable' in dist_names:
    dist_names.remove('levy_stable')
    


dist_names_short = ['beta',
              'expon',
              'gamma',
              'lognorm',
              'norm',
              'pearson3',
              'triang',
              'uniform',
              'weibull_min', 
              'weibull_max']

# Set up empty lists to stroe results
chi_square = []
p_values = []

# Set up 50 bins for chi-square test
# Observed data will be approximately evenly distrubuted aross all bins
percentile_bins = np.linspace(0,100)



percentile_cutoffs = np.percentile(y_std, percentile_bins)
observed_frequency, bins = (np.histogram(y_std, bins=percentile_cutoffs))
cum_observed_frequency = np.cumsum(observed_frequency)


all_distributions_df = pd.DataFrame(columns = dist_names)



print('Fitting Distribution to Target')









# Loop through candidate distributions
def get_chi_square_and_ks_test(all_distributions_df):   
    # Get histogram of original data
    y, x = np.histogram(y_std, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    results = pd.DataFrame()

    for idx, distribution in enumerate(all_distributions_df.columns):
#        print('>>>>' + distribution)

        # Set up distribution and get fitted distribution parameters
        dist = getattr(scipy.stats, distribution)
        param = dist.fit(y_std)
        
        # Obtain the KS test P statistic, round it to 5 decimal places
        p = scipy.stats.kstest(y_std, distribution, args=param)[1]
        p = np.around(p, 5)
#        p_values.append(p)    
        
        # Get expected counts in percentile bins
        # This is based on a 'cumulative distrubution function' (cdf)
        cdf_fitted = dist.cdf(percentile_cutoffs, *param[:-2], loc=param[-2], 
                              scale=param[-1])
        expected_frequency = []
        for bin in range(len(percentile_bins)-1):
            expected_cdf_area = cdf_fitted[bin+1] - cdf_fitted[bin]
            expected_frequency.append(expected_cdf_area)
        
        # calculate chi-squared
        expected_frequency = np.array(expected_frequency) * size
        cum_expected_frequency = np.cumsum(expected_frequency)
        ss = sum (((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency)
#        chi_square.append(ss)
            
        # Collate results and sort by goodness of fit (best at top)
#        results['Distribution'] = distribution
#        results['chi_square'] = ss
#        results['p_value'] = p
        
        
        
        results.loc[distribution, 'Distribution'] = distribution
        results.loc[distribution, 'chi_square'] = ss
        results.loc[distribution, 'p_value'] = p
        
        
        
        
        #other code
        # fit dist to data
#        params = dist.fit(y_std)

        # Separate parts of parameters
        arg = param[:-2]
        loc = param[-2]
        scale = param[-1]

        # Calculate fitted PDF and error with fit in distribution
        pdf = dist.pdf(x, loc=loc, scale=scale, *arg)
        sse = np.sum(np.power(y - pdf, 2.0))

        results.loc[str(distribution), 'sse'] = sse
        
        
        
#        
#        #my code
#        statistic, critical_values, significance_level = scipy.stats.anderson(y_std, dist = dist)        
#        results.loc[distribution, 'ad_test'] = statistic
        
        
        
                
                
        print(distribution)

        
    return results
        
        


#results = get_chi_square_and_p_value(all_distributions_df)

from ParallelCPU import ParallelCPU
#results = parallise(all_distributions_df, function = get_chi_square_and_ks_test)

parallel = ParallelCPU(debug_mode = False)
results = parallel.compute(all_distributions_df, function = get_chi_square_and_ks_test)

# Report results

print ('\nDistributions sorted by goodness of fit:')
print ('----------------------------------------')
results_new = results
results.sort_values(['p_value'], ascending = False, inplace=True)

print (results)




#results_original = results
#results_original.index = results_original['Distribution']
#results_original = results_original.drop(columns = ['Distribution'])


# Divide the observed data into 100 bins for plotting (this can be changed)
number_of_bins = 100
bin_cutoffs = np.linspace(np.percentile(y,0), np.percentile(y,99),number_of_bins)

# Create the plot
h = plt.hist(y, bins = bin_cutoffs, color='0.75')
plt.show()


# Get the top three distributions from the previous phase
number_distributions_to_plot = 10
dist_names = results['Distribution'].iloc[0:number_distributions_to_plot]

# Create an empty list to stroe fitted distribution parameters
parameters = []

# Loop through the distributions ot get line fit and paraemters


#
#
#plt.figure()
#plt.title('Target - Normal Distribution')
#sns.distplot(y_train, fit=stats.norm, color = 'r', kde = True, rug = True)
#



for idx, dist_name in enumerate(dist_names):
    # Set up distribution and store distribution paraemters
    dist = getattr(scipy.stats, dist_name)
    param = dist.fit(y_std)
    parameters.append(param)
    

    plt.figure()
    plt.title('Target - ' + dist_name)
    sns.distplot(y_std, fit=dist, color = 'r', kde = True, rug = False, hist = True)
    plt.show()
        
























# Add legend and display plot
#
#plt.legend()
#plt.show()

# Store distribution paraemters in a dataframe (this could also be saved)
dist_parameters = pd.DataFrame()
dist_parameters['Distribution'] = (
        results['Distribution'].iloc[0:number_distributions_to_plot])
dist_parameters['Distribution parameters'] = parameters

# Print parameter results
print ('\nDistribution parameters:')
print ('------------------------')

for index, row in dist_parameters.iterrows():
    print ('\nDistribution:', row[0])
    print ('Parameters:', row[1] )
    
    
    
    
    


## qq and pp plots
    
data = y_std.copy()
data.sort()

# Loop through selected distributions (as previously selected)

for distribution in dist_names:
    # Set up distribution
    dist = getattr(scipy.stats, distribution)
    param = dist.fit(y_std)
    
    # Get random numbers from distribution
    norm = dist.rvs(*param[0:-2],loc=param[-2], scale=param[-1],size = size)
    norm.sort()
    
    # Create figure
    fig = plt.figure(figsize=(8,5)) 
    
    # qq plot
    ax1 = fig.add_subplot(121) # Grid of 2x2, this is suplot 1
    ax1.plot(norm,data,"o")
    min_value = np.floor(min(min(norm),min(data)))
    max_value = np.ceil(max(max(norm),max(data)))
    ax1.plot([min_value,max_value],[min_value,max_value],'r--')
    ax1.set_xlim(min_value,max_value)
    ax1.set_xlabel('Theoretical quantiles')
    ax1.set_ylabel('Observed quantiles')
    title = 'qq plot for ' + distribution +' distribution'
    ax1.set_title(title)
    
    # pp plot
    ax2 = fig.add_subplot(122)
    
    # Calculate cumulative distributions
    bins = np.percentile(norm,range(0,101))
    data_counts, bins = np.histogram(data,bins)
    norm_counts, bins = np.histogram(norm,bins)
    cum_data = np.cumsum(data_counts)
    cum_norm = np.cumsum(norm_counts)
    cum_data = cum_data / max(cum_data)
    cum_norm = cum_norm / max(cum_norm)
    
    # plot
    ax2.plot(cum_norm,cum_data,"o")
    min_value = np.floor(min(min(cum_norm),min(cum_data)))
    max_value = np.ceil(max(max(cum_norm),max(cum_data)))
    ax2.plot([min_value,max_value],[min_value,max_value],'r--')
    ax2.set_xlim(min_value,max_value)
    ax2.set_xlabel('Theoretical cumulative distribution')
    ax2.set_ylabel('Observed cumulative distribution')
    title = 'pp plot for ' + distribution +' distribution'
    ax2.set_title(title)
    
    # Display plot    
    plt.tight_layout(pad=4)
    plt.show()
    
    
    
    
    
    