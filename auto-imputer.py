

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline

from sklearn import ensemble
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

#from sklearn.pipeline import Pipeline

#
#
#class pre_processing_transformer(BaseEstimator, TransformerMixin):
##    def __init__(self, categorical_columns, continuous_columns, enable_ohe_for_categorical = False, exclude_column_from_ohe = None, imputer = 'miss_forest'):
#    def __init__(self):
#
#        
#        #        self.categorical_columns = categorical_columns
##        self.continuous_columns = continuous_columns
##        self.enable_ohe_for_categorical = enable_ohe_for_categorical
##        self.exclude_column_from_ohe = exclude_column_from_ohe
#        
#        self.imputer_obj = None
#        self.x_columns = None
#        
#        self.count = 0
#       
#    def fit(self, x, y=None):
#        joint_df = x
#        #get indexes of categorical columns. this is used by miss_forest to identify categorical features
##        cat_column_indexes_in_joint_df = []    
##        for column in self.categorical_columns:
##            cat_column_indexes_in_joint_df.append(joint_df.columns.get_loc(column))
## 
##        cat_column_indexes_in_joint_df = np.asarray(cat_column_indexes_in_joint_df)
#        
#        
#        print('Using random forest as imputer')
#        imputer = MissForest(max_iter = 1, n_estimators = 2, n_jobs = 24, verbose = 0)
##        imputer.fit(joint_df, cat_vars = cat_column_indexes_in_joint_df )
#        imputer.fit(joint_df )
#
#
#        self.imputer_obj = imputer
#      
#        return self
#    
#    
#    def transform(self, x, y=None):
#        #imputation leads to loss of column information. this step restores column names and gives back dataframe
##        print(self.count)
#        self.count = self.count+1
#        imputer = self.imputer_obj
#        imputed_df = imputer.transform(x)
##        imputed_df = pd.DataFrame(imputed_df, columns = self.x_columns )
#        return imputed_df
#        





         
        



#def pre_processing(imputer, enable_ohe, exclude_column_from_ohe):   
#    
#
#        
#    # remove features with zero varience
#    selector = feature_selection.VarianceThreshold()
#    selector.fit(imputed_df)
#    
#    supports = selector.get_support()
#    indices_with_zero_variance = np.where(supports == False)
#    indices_with_zero_variance = list(indices_with_zero_variance[0])    
#    columns_to_drop = imputed_df.columns[indices_with_zero_variance]
#    
#    
#    cat_columns_list_after_feature_removal = list(cat_columns)
#    cont_columns_list_after_feature_removal = list(cont_columns)
#    
#    
#    for column in columns_to_drop:
#        drop_and_log_column(imputed_df, column, '_Zero variance after imputation')
#        warn(str(column) + ' Dropped column due to zero varience after imputation. Maybe its better to make missing values as unknown')
#        
#        if column in cat_columns_list_after_feature_removal:
#            cat_columns_list_after_feature_removal.remove(column)
#        elif column in cont_columns_list_after_feature_removal:
#            cont_columns_list_after_feature_removal.remove(column)
#    
#    
#    if show_plots:
#        show_heatmap(imputed_df, message = 'heatmap_after_imputation')
#        
#
#    
#    
#    if enable_ohe:
#        #the cat and cont are seperated and then joint so that the order is preserved after one hot
#        imputed_cat_df = imputed_df[list(cat_columns_list_after_feature_removal)]
#        imputed_cont_df = imputed_df[list(cont_columns_list_after_feature_removal)]
#        
#        
#        #one hot categorical columns    
#        imputed_cat_df = one_hot(imputed_cat_df, exclude_column_from_ohe)
#        
#        imputed_df = imputed_cat_df.join(imputed_cont_df,how = 'outer')
#
#    
#    
#    return imputed_df
#    
#
#
#
#
#categorical_df = combined_categorical_df
#continuous_df = combined_continuous_df
#
#
#    
#categorical_df_label_encoded =  label_encoder(categorical_df, strategy = 'keep_missing_as_nan')
##here outer is used. so continuous
#cat_columns = categorical_df_label_encoded.columns
#cont_columns = continuous_df.columns
#
#joint_df = categorical_df_label_encoded.join( continuous_df, how = 'outer')      
#
#
#imputer_transformer = pre_processing_transformer()
#
#
#rf_estimator = ensemble.RandomForestRegressor()
#
#
#
#
##pipe = make_pipeline(imputer_transformer, rf_estimator)
#
#
#steps =[ 
#            ('imputer', imputer_transformer), 
#            ('rfe', rf_estimator) 
#        ]
#
#pipe = Pipeline(steps)
#
#
#param_grid = {
#            'rfe__n_estimators' : get_equally_spaced_numbers_in_range(1,10000) ,
#            'rfe__max_depth' : [1,3,5,10],
#            'rfe__max_features': ['sqrt', 'log2'],     
#            
#        }
#
#
#pipe_grid_estimator = model_selection.GridSearchCV(pipe, param_grid, cv=5)
#
#
#pipe_grid_estimator.fit(joint_df.values, y_train)
#
#
#sorted(pipe.get_params().keys())
#
#
#
#
#imputer_transformer.fit(joint_df.values)
#t = imputer_transformer.transform(joint_df.values)
#
#regr = ensemble.RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
#regr.fit(t,y_train)
#regr.predict(t)
#
#
#
#
#if show_plots:
#    show_heatmap(joint_df, message = 'heatmap_before_imputation')
#
#
#



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# To use this experimental feature, we need to explicitly ask for it:
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score



categorical_df = combined_categorical_df
continuous_df = combined_continuous_df


categorical_df_label_encoded =  label_encoder(categorical_df, strategy = 'keep_missing_as_nan')
#here outer is used. so continuous
cat_columns = categorical_df_label_encoded.columns
cont_columns = continuous_df.columns

joint_df = categorical_df_label_encoded.join( continuous_df, how = 'outer')      



joint_df = categorical_df_label_encoded.join( continuous_df, how = 'outer')      


estimators = [
    BayesianRidge(),
    DecisionTreeRegressor(max_features='sqrt', random_state=0),
    ExtraTreesRegressor(n_estimators=100, max_depth = 1, random_state=0),    
    KNeighborsRegressor(n_neighbors=4)
]


random_forest_trees = 1000
rf_estimator = ensemble.RandomForestRegressor(n_estimators = random_forest_trees, max_depth = 1)


from xgboost import XGBRegressor





#test_estimator = BayesianRidge()
#test_estimator = rf_estimator

test_estimator = XGBRegressor()



    
N_SPLITS = 10

X_missing = joint_df.values
y_missing = y_train


score_iterative_imputer = pd.DataFrame()


# Estimate the score after imputation (mean and median strategies)
score_simple_imputer = pd.DataFrame()
for strategy in ('mean', 'median'):
    estimator = make_pipeline(
        SimpleImputer(missing_values=np.nan, strategy=strategy),
        test_estimator
    )
    score_simple_imputer[strategy] = cross_val_score(
        estimator, X_missing, y_missing, scoring='neg_mean_squared_error',
        cv=N_SPLITS, n_jobs=-1
    )



count = 0
for impute_estimator in estimators:
    print(str(count))
    count = count + 1
    estimator = make_pipeline(
        IterativeImputer(random_state=0, estimator=impute_estimator),
        test_estimator
    )
    score_iterative_imputer[impute_estimator.__class__.__name__] = \
        cross_val_score(
            estimator, X_missing, y_missing, scoring='neg_mean_squared_error',
            cv=N_SPLITS, n_jobs=-1
        )

scores = pd.concat(
    [score_simple_imputer, score_iterative_imputer],
    keys=['Original', 'SimpleImputer', 'IterativeImputer'], axis=1
)



# plot boston results
fig, ax = plt.subplots(figsize=(13, 6))
means = -scores.mean()
errors = scores.std()
means.plot.barh(xerr=errors, ax=ax)
ax.set_title('Housing Regression with Different Imputation Methods' + '  Baysian Regressor: ' + str(random_forest_trees))
ax.set_xlabel('MSE (smaller is better)')
ax.set_yticks(np.arange(means.shape[0]))
ax.set_yticklabels([" w/ ".join(label) for label in means.index.get_values()])
plt.tight_layout(pad=1)
plt.show()



















































    
    