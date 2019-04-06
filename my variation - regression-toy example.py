import os
import pydot
import io
import pandas as pd
import seaborn as sns
import numpy as np
import math
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing, tree, svm, neighbors, metrics, linear_model, manifold, linear_model, ensemble
from sklearn_pandas import DataFrameMapper,CategoricalImputer
from sklearn import model_selection, metrics, ensemble, preprocessing, decomposition, feature_selection
from sklearn.model_selection import validation_curve, train_test_split
from sklearn.datasets import make_classification
from sklearn.datasets import make_circles, make_blobs, make_moons, make_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

def rmse(y_orig, y_pred):
    return math.sqrt(metrics.mean_squared_error(y_orig,y_pred) )

def generate_data1(n_samples):
    np.random.seed(0)
    X = np.random.rand(n_samples)
    y = np.cos(2 * np.pi * X) + np.random.randn(n_samples) * 0.1
    return X.reshape(-1,1), y

def generate_data2():
    X = np.array([0.1,0.13,0.2, 0.22,0.25, 0.3, 0.5, 0.6, 0.65, 0.7 ])
    y = np.array([0.9,0.88, 1, 0.5, 0.6, 0.5, -0.5, -0.4, -0.3, -0.42 ])
    return X.reshape(-1, 1), y

def generate_data(n_samples, n_features):
    np.random.seed(0)
    X = 6 * np.random.rand(n_samples, n_features)
    if n_features == 1:
        y = np.sin(X).ravel()
        y[::5] += 2 * (0.5 - np.random.rand((int)(n_samples/5)) )
    elif n_features == 2:
        y = 0.5 + 1.5 * X[:,0]**2 + 1.6 * X[:,1]**3
    return X, y

def plot_data_2d(X, y, limit_y=None, message = ''):
    plt.figure()
    labels =['X'+str(i) for i in range(X.shape[1])]
    if limit_y:
        plt.ylim(limit_y[0], limit_y[1])
    plt.scatter(X, y, c='blue')
    plt.xlabel(labels[0])
    plt.ylabel("target")
    plt.title(message, fontdict = {'fontsize' : 20})
    plt.show()

def plot_model_2d(estimator, X, y, limit_y=None, message = ''):
    plt.figure()
    labels =['X'+str(i) for i in range(X.shape[1])]

    plt.scatter(X, y, c='blue')
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    xx = np.arange(x_min, x_max, 0.1)
    y_pred = estimator.predict(xx.reshape(-1, 1))
    if limit_y:
        plt.ylim(limit_y[0], limit_y[1])
    plt.plot(xx, y_pred, color='red')
    plt.title(message, fontdict = {'fontsize' : 20})
    plt.show()

def plot_data_3d(X, y, message = ''):
    labels =['X'+str(i) for i in range(X.shape[1])]
    fig = plt.figure()
    ax = plt.axes(projection='3d')    
    ax.scatter(X[:, 0], X[:, 1], y, s=30, c = 'grey')  
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel("target")
    plt.title(message, fontdict = {'fontsize' : 20})
    
#    for angle in range(0, 360):
#        ax.view_init(30, angle)
#        plt.draw()
#        plt.pause(.01)
    plt.show()

def plot_model_3d(estimator, X, y, message = ''):
    labels =['X'+str(i) for i in range(X.shape[1])]

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d') 
    ax.scatter(X[:,0], X[:,1], y, c = 'grey', s=30)
    ax.plot_surface(xx, yy, Z)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel("target")
    plt.title(message, fontdict = {'fontsize' : 20})
    plt.tight_layout()    
    
#    for angle in range(0, 360):
#        ax.view_init(20, angle)
#        plt.draw()
#        plt.pause(.01)        
    plt.show()

def plot_residuals(estimator, X, y):
    y_pred = estimator.predict(X)
    error = y_pred - y
    plt.scatter(y_pred, error, c='blue', marker='o', label='Training data') 
    xmin, xmax = y_pred.min(), y_pred.max()
    plt.hlines(y=0, xmin=xmin, xmax=xmax, lw=2, color='red')

def get_model_objective(estimator, grid, X, y):
    grid_estimator = model_selection.GridSearchCV(estimator, grid, scoring=metrics.make_scorer(rmse, greater_is_better=False), cv=10, n_jobs = -1)
    grid_estimator.fit(X, y)
    print(grid_estimator.best_params_)
    final_model = grid_estimator.best_estimator_
    #print(final_model.coef_)
    #print(final_model.intercept_)
    print(grid_estimator.best_score_)
    print(grid_estimator.score(X, y))
    return final_model

def get_model_neighbors(estimator, grid, X, y):
    grid_estimator = model_selection.GridSearchCV(estimator, grid, scoring=metrics.make_scorer(rmse, greater_is_better=False), cv=10, n_jobs = -1)
    grid_estimator.fit(X_train, y_train)
    print(grid_estimator.best_params_)
    final_model = grid_estimator.best_estimator_
    print(grid_estimator.best_score_)
    print(grid_estimator.score(X_train, y_train))
    return final_model

def write_to_pdf(estimator, X, path):
    dot_data = io.StringIO() 
    tree.export_graphviz(estimator, out_file = dot_data, feature_names = X.columns)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
    graph.write_pdf(path)

def get_model_tree(estimator, grid, X, y, path="C://tree.pdf"):
    grid_estimator = model_selection.GridSearchCV(estimator, grid, scoring=metrics.make_scorer(rmse, greater_is_better=False), cv=10, n_jobs = -1)
    labels =['X'+str(i) for i in range(X.shape[1])]
    tmp_df = pd.DataFrame(X, columns=labels)
    grid_estimator.fit(tmp_df, y)
    print(grid_estimator.best_params_)
    final_model = grid_estimator.best_estimator_
    #write_to_pdf(final_model, tmp_df, path)
    print(grid_estimator.best_score_)
    print(grid_estimator.score(tmp_df, y))
    return final_model

def get_model_ensemble(estimator, grid, X, y):
    grid_estimator = model_selection.GridSearchCV(estimator, grid, scoring=metrics.make_scorer(rmse, greater_is_better=False), cv=10, n_jobs = -1)
    grid_estimator.fit(X, y)
    print(grid_estimator.best_params_)
    final_model = grid_estimator.best_estimator_
    print(grid_estimator.best_score_)
    print(grid_estimator.score(X, y))
    return final_model

def grid_search_one_parameter(estimator, grid, X, y, scoring="accuracy", message = ''):
    plt.figure()
    grid_estimator = model_selection.GridSearchCV(estimator, grid, cv=10, return_train_score=True, scoring = scoring, n_jobs = -1)
    grid_estimator.fit(X, y)

    train_mean = grid_estimator.cv_results_.get('mean_train_score')
    train_std = grid_estimator.cv_results_.get('std_train_score')
    test_mean = grid_estimator.cv_results_.get('mean_test_score')
    test_std = grid_estimator.cv_results_.get('std_test_score')

    plt.style.use('seaborn')

    param_name = list(grid.keys())[0]
    param_range = grid.get(param_name)
    # Plot mean accuracy scores for training and test sets
    plt.plot(param_range, train_mean, label="Training score", color="black")
    plt.plot(param_range, test_mean, label="Cross-validation score", color="red")

    # Plot accurancy bands for training and test sets
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

    # Create plot
    plt.title("Parameter Values vs Performance:" +  str(estimator).split('(')[0] + ' model')
    plt.xlabel(param_name)
    plt.ylabel("Performance")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.title('Grid Search: ' + message, fontdict = {'fontsize' : 20})
    plt.show()
    
def grid_search_two_parameters(estimator, grid, X, y, scoring="accuracy", message = ''):
    grid_estimator = model_selection.GridSearchCV(estimator, grid, cv=10, scoring = scoring, return_train_score=True, n_jobs = -1)
    grid_estimator.fit(X, y)

    param1_name = list(grid.keys())[0]
    param1_range = grid.get(param1_name)    
    param2_name = list(grid.keys())[1]
    param2_range = grid.get(param2_name)
        
    plt.figure()
    ax = plt.axes(projection='3d')
    X, Y = np.meshgrid(param1_range, param2_range)
    
    Z = grid_estimator.cv_results_.get('mean_test_score').reshape(X.shape)
    ax.plot_surface(X, Y, Z, color='red', label="Cross-validation score")
    
    Z = grid_estimator.cv_results_.get('mean_train_score').reshape(X.shape)
    ax.plot_surface(X, Y, Z, color='black', label="Training score")
    
    ax.set_xlabel(param1_name)
    ax.set_ylabel(param2_name)
    ax.set_zlabel('performance')
#    plt.legend(loc="best")
    plt.tight_layout()
    plt.title('Grid Search: ' + message, fontdict = {'fontsize' : 20})
    
   
#    for angle in range(0, 180):
#        ax.view_init(20, angle)
#        plt.draw()
#        plt.pause(.01)
    plt.show()


#linear pattern in 2d
message = 'Linear Pattern in 2d'
X, y = make_regression(n_samples=200, n_features=1, n_informative=1, random_state=0, noise=20, bias = 250)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
plot_data_2d(X_train, y_train, message = message)



message = 'Linear Regression'
linear_estimator = linear_model.LinearRegression()
linear_grid = {'normalize': [True, False]}
final_estimator = get_model_objective(linear_estimator, linear_grid, X_train, y_train)
plot_model_2d(final_estimator, X_train, y_train, message = message)



#linear pattern in 3d
message = 'Data - linear pattern in 3d'
X, y = make_regression(n_samples=200, n_features=2, n_informative=2, random_state=0, noise=0, bias = 250)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
plot_data_3d(X_train, y_train, message = message)



message = 'Linear Regression'
linear_estimator = linear_model.LinearRegression()
linear_grid = {'normalize': [True, False]}
final_estimator = get_model_objective(linear_estimator, linear_grid, X_train, y_train)
plot_model_3d(final_estimator, X_train, y_train, message = message)


#non-linear pattern in 2d
message = 'Data - Non-Linear pattern in 2d'
X, y = generate_data(n_samples=100, n_features=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
plot_data_2d(X_train, y_train, message = message)


message = 'Linear Regression 2D'
linear_estimator = linear_model.LinearRegression()
linear_grid = {'normalize': [True, False]}
final_estimator = get_model_objective(linear_estimator, linear_grid, X_train, y_train)
plot_model_2d(final_estimator, X_train, y_train, message = message)


message = 'Poly Estimator 2D'
poly_estimator = Pipeline([('features', PolynomialFeatures()) ,
                          ('estimator', linear_model.LinearRegression())]
                          )
poly_grid = {'features__degree':list(range(1,15))}
final_estimator = get_model_objective(poly_estimator, poly_grid, X_train, y_train)
grid_search_one_parameter(poly_estimator, poly_grid, X_train, y_train, scoring = metrics.make_scorer(rmse), message = message)
plot_model_2d(final_estimator, X_train, y_train, message = message)


message = 'Decision Tree Regressor'
dt_estimator = tree.DecisionTreeRegressor()
dt_grid = {'max_depth':list(range(1,10))}
final_estimator = get_model_tree(dt_estimator, dt_grid, X_train, y_train)
grid_search_one_parameter(dt_estimator, dt_grid, X_train, y_train, scoring = metrics.make_scorer(rmse))
plot_model_2d(final_estimator, X_train, y_train, message = message)


message = 'Random Forest Regressor'
rf_estimator = ensemble.RandomForestRegressor()
rf_grid = {'n_estimators':list(range(1,100,20)), 'max_depth':list(range(1,5))}
final_estimator = get_model_ensemble(rf_estimator, rf_grid, X_train, y_train)
grid_search_two_parameters(rf_estimator, rf_grid, X_train, y_train, scoring = metrics.make_scorer(rmse), message = message)
plot_model_2d(final_estimator, X_train, y_train, message = message)



message = 'Linear SVM'
svr_lin = svm.SVR(kernel='linear')
svr_lin_grid = {'C':[20, 50, 100] }
final_estimator = get_model_objective(svr_lin, svr_lin_grid, X_train, y_train)
grid_search_one_parameter(svr_lin, svr_lin_grid, X_train, y_train, scoring = metrics.make_scorer(rmse))
plot_model_2d(final_estimator, X_train, y_train, message = message)


message = 'Kernel SVR (RBF)'
svr_rbf = svm.SVR(kernel='rbf')
svr_rbf_grid = {'C':[50, 100], 'gamma':[0.1, 0.2] }
final_estimator = get_model_objective(svr_rbf, svr_rbf_grid, X_train, y_train)
grid_search_two_parameters(svr_rbf, svr_rbf_grid, X_train, y_train, scoring = metrics.make_scorer(rmse), message = message)
plot_model_2d(final_estimator, X_train, y_train, message = message)


message = 'Kernel SVR (Poly)'
svr_poly = svm.SVR(kernel='poly')
svr_poly_grid = {'C':[50, 100], 'degree':[2, 3] }
final_estimator = get_model_objective(svr_poly, svr_poly_grid, X_train, y_train)
grid_search_two_parameters(svr_poly, svr_poly_grid, X_train, y_train, scoring = metrics.make_scorer(rmse), message = message)
plot_model_2d(final_estimator, X_train, y_train, message = message)


message = 'KNN Regressor'
knn_estimator = neighbors.KNeighborsRegressor()
knn_grid = {'n_neighbors':[8] }
final_estimator = get_model_neighbors(knn_estimator,knn_grid, X_train, y_train)
grid_search_one_parameter(knn_estimator, knn_grid, X_train, y_train, scoring = metrics.make_scorer(rmse), message = message)
plot_model_2d(final_estimator, X_train, y_train, message = message)



#non-linear pattern in 3d
message = 'Non-Linear Pattern in 3d'
X, y = generate_data(n_samples=100, n_features=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
plot_data_3d(X_train, y_train, message = message)



message = 'Linear Regression in 3d'
linear_estimator = linear_model.LinearRegression()
linear_grid = {'normalize': [True, False]}
final_estimator = get_model_objective(linear_estimator, linear_grid, X_train, y_train)
plot_model_3d(final_estimator, X_train, y_train, message = message)


message = 'KNN Regression in 3d'
knn_estimator = neighbors.KNeighborsRegressor()
knn_grid = {'n_neighbors':list(range(1,10)) }
final_estimator = get_model_neighbors(knn_estimator,knn_grid, X_train, y_train)
grid_search_one_parameter(knn_estimator, knn_grid, X_train, y_train, scoring = metrics.make_scorer(rmse), message = message)
plot_model_3d(final_estimator, X_train, y_train, message = message)


message = 'SVM Kernel (RBF) 3d'
svr_rbf = svm.SVR(kernel='rbf')
svr_rbf_grid = {'C':[50, 100], 'gamma':[0.1, 0.2] }
final_estimator = get_model_objective(svr_rbf, svr_rbf_grid, X_train, y_train)
grid_search_two_parameters(svr_rbf, svr_rbf_grid, X_train, y_train, scoring = metrics.make_scorer(rmse), message = message)
plot_model_3d(final_estimator, X_train, y_train, message = message)

