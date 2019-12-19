#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" 
This library provides a basic set of tools to augment a dataset with basic statistics,  
perform recursive feature elimination and hyperparameter tuning for a set of pre-defined 
regression models commonly used in machine learning.
"""
#------------------------------------------------------------------------------------------------------
# importing "copy" for copy operations 
from copy import deepcopy  #Example of deep copy:    b = deepcopy(a)
                           #Example of shallow copy: b = copy.copy(a)
             
import numpy as np         #To update numpy type:  sudo -H pip3 install --upgrade numpy

import json

import pandas as pd            #Quick summary: https://pandas.pydata.org/pandas-docs/stable/10min.html
#import statsmodels.api as sm  #sudo apt-get install python3-statsmodels

#Note to install scikit learn: sudo -H pip3 install -U scikit-learn
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.neighbors.kde import KernelDensity
from scipy.stats import iqr

import pickle          #This library is to store objects in disk and load objects from disk.


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge 
from sklearn.linear_model import Lasso
from sklearn.linear_model import BayesianRidge
from sklearn.tree     import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import  GradientBoostingRegressor
from sklearn.svm      import SVR
from sklearn.svm      import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import LogisticRegression
#The following line is useful for testing the Jupyter notebook.
#%matplotlib inline 

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#Classes 
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
        
class prediction_class:
    """ 
    This class to store results of each model after the hyperparameter search.
    
    Attributes
    ----------
    pipeline_name : str
        A descriptive name for the model used in the pipeline.
    best_pipeline : Pipeline
        Best pipeline: This includes scaling, estimator, etc (this is what 
        you should use when calling predict).
    grid_search_flag : bool
        True if the the hyperparameters were tuned with a grid search, False 
        otherwise.
    best_estimator_model : sklearn estimator
        This contains only the estimator, it does not contain any additional 
        steps of the pipeline such as the scaler.
    tuned_hyperparameters : dict
        Hyperparameters that were tuned for the the best estimator model 
        (this field contains information only if grid_search_flag is True).
    all_hyperparameters : dict
        All the hyperparameters that characterize the best estimator model. 
    names_of_optimal_features : list
        Names of features used by the model, represented as a list of strings. 
    performance_metric_value : numeric
        Value of the calculated performance metric.
    performance_metric_name : str
        Name of the performance metric used.
    confusion_matrix : numpy.ndarray
        Confusion matrix for the selected model. Only available when used
        on classification.
    classification_report : str
        Report with the main classification metrics. Only available when used
        on classification.
    test_rows : list
        List of the indexes of the rows used as the test set.
    """

    def __init__(self, pipeline_name='', best_pipeline=[],
                 grid_search_flag=False, best_estimator_model=None,
                 tuned_hyperparameters={}, all_hyperparameters={},
                 names_of_optimal_features=[], performance_metric_value=0.0,
                 performance_metric_name='', confusion_matrix=None,
                 classification_report='', test_rows=[]):
        self.pipeline_name                 = pipeline_name
        self.best_pipeline                 = deepcopy(best_pipeline)
        self.grid_search_flag              = grid_search_flag
        self.best_estimator_model          = deepcopy(best_estimator_model)
        self.tuned_hyperparameters         = deepcopy(tuned_hyperparameters)
        self.all_hyperparameters           = deepcopy(all_hyperparameters)
        self.names_of_optimal_features     = deepcopy(names_of_optimal_features)
        self.performance_metric_value      = performance_metric_value
        self.performance_metric_name       = performance_metric_name
        self.confusion_matrix              = confusion_matrix
        self.classification_report         = classification_report
        self.test_rows                     = deepcopy(test_rows)

def print_results_for_tested_prediction_models(p,extra_title_str=''):
    """
    This auxiliar function prints some basic results from the regression 
    models that were trained using grid-search and cross validation.
    
    Parameters
    ----------
    p : list
        List with objects of the regression_class. 
    extra_title_string: Character string that is added 
        to "Prediction performance".
    
    Returns
    -------
    None
    """
    print('____________________________________________________________________________________________')
    print('Prediction performance %s ' % extra_title_str)
    print('____________________________________________________________________________________________')
    for idx in range(len(p)):
        if idx != 0 : print('\n',end='')
        print('%s = %.2f; %s.' % (p[idx].performance_metric_name,
                                      p[idx].performance_metric_value,
                                      p[idx].pipeline_name))
        if  p[idx].grid_search_flag ==True:
            print('Best parameters: %s.'
                  % (p[idx].tuned_hyperparameters))
    print('____________________________________________________________________________________________')
    print('\n\n')
    

def check_score(score, model_type):
    """
    Check if the selected score is suitable for the model type.
    
    Parameters
    ------------
    score : str
        Name of the score.
    model_type : str
        Type of the model, it could be 'regression' or 'classification'.
    
    Returns
    -------
    None
    """
    
    regression_scores = ['explained_variance', 'neg_mean_absolute_error',
                          'neg_mean_squared_error','neg_median_absolute_error',
                          'neg_mean_squared_log_error','r2']
    
    classification_scores = ['accuracy', 'balanced_accuracy',
                             'average_precision', 'brier_score_loss', 'f1',
                             'f1_micro', 'f1_macro', 'f1_weighted',
                             'f1_samples', 'neg_log_loss', 'precision',
                             'precision_micro','precision_macro',
                             'precision_weighted', 'recall', 'recall_micro',
                             'recall_macro', 'recall_weighted', 'roc_auc']
    if model_type=='regression':
        if score not in regression_scores:
            raise Exception('Score %s is not a regression score' % (score))
    
    elif model_type=='classification':
        if score not in classification_scores:
            raise Exception('Score %s is not a classification score' % (score))
    else:
        raise Exception('%s is not a valid type of model' % (model_type))


def check_split_type(split_type):
    """Check if te split type is a valid one."""
    types = ['simple','stratified']
    if split_type not in types:
        raise Exception('%s is not a valid split type' % (split_type))
        

def check_model_type(predictor, model_type):
    """
    Check if the predictor has the correct type
    
    Parameters
    ------------
    score : str
        The selected predictor.
    model_type : str
        Type of the model, it could be 'regression' or 'classification'.
    
    Returns
    -------
    None
    """
    regressors = ['LinearRegression','Ridge','Lasso','BayesianRidge',
                  'DecisionTreeRegressor','RandomForestRegressor','SVR',
                  'GradientBoostingRegressor','MLPRegressor']
    
    classifiers = ['RandomForestClassifier','ExtraTreesClassifier','SVC',
                    'MLPClassifier', 'MultinomialNB']
            
    if model_type=='regression':
        if predictor not in regressors:
            raise Exception('Model %s is not a regression model' % (predictor))
    
    elif model_type=='classification':
        if predictor not in classifiers:
            raise Exception('Model %s is not a classification model' %(predictor))
    else:
        raise Exception('%s is not a valid type of model' % (model_type))


def dataframe_split(df_x,df_y,percentage_for_testing,split_type):
    """
    This function splits two datasets with the same number of observations
    to create test and training dataframes.
    
    Parameters
    ----------
    df_x : Dataframe
        Dataframe with input data
    df_y : Dataframe
        Dataframe with output data
    percentage_for_testing : numeric
        Percentage of the data the will be used for_testing
    split_type : str
        It can be either 'simple' or 'stratified'.
    
    Returns
    -------
    DataFrame, DataFrame, DataFrame, DataFrame:
        Four Dataframe in the following order: Dataframe with input data for 
        training, Dataframe with output data for training, Dataframe with input 
        data for testing, Dataframe with output data for testing.
    """
    check_split_type(split_type)
    rows = []
    if len(df_x.index) != len(df_y.index):
        raise Exception('df_x and df_y should have the same number of observations (rows)')
    elif split_type=='simple':
        num_observations              = len(df_x)
        #Casting to int
        num_observations_for_test_set = \
                        int(np.round(num_observations*percentage_for_testing))
        #Extract a few random indices
        rows =list(np.random.randint(num_observations, size=\
                                     num_observations_for_test_set))
        
    elif split_type=='stratified':
        #Get the classification labels
        labels = np.unique(df_y.iloc[:,0])
        dicty = {}
        for x in labels: dicty[x] = []
        
        #df_y - [1,2,4,5,4,2,3,4,1,2]
        #Find where each label is in the data frame
        for index in range(len(df_y)):
            label = df_y.iloc[index,0]
            dicty[label].append(index)
        
        rows = []
        #For each kind of label create a random subset to be in the training 
        #set
        for label in labels:
            num_observations = len(dicty[label])
            #Casting to int
            num_observations_test = int(np.round(num_observations*\
                                                     percentage_for_testing))
            
            test_list = np.random.choice(dicty[label],size= \
                                         num_observations_test,replace=False)
            
            rows = rows + list(test_list)
        
    #Extract test set.
    df_x_test  = df_x.iloc[rows,:]
    #The rest is the train set.                                      
    df_x_train = df_x.drop(df_x.index[rows])                                                         

    df_y_test  = df_y.iloc[rows,:]
    df_y_train = df_y.drop(df_y.index[rows])

    return df_x_train,df_x_test,df_y_train,df_y_test,rows


def get_optimal_features_for_each_model(p,df_X,df_y,scoring,
                                        features_to_eliminate_per_step=1,
                                        k_folds=5,verbose=True,
                                        split_type='simple'):
    #Note: either coef_ or feature_importances_ attributes are needed by the  
    #RFECV funciton to work. 
    #0 LinearRegression:             coef_
    #1 Ridge regression:             coef_
    #2 Lasso regression:             coef_
    #3 Bayesian Ridge:               coef_
    #4 Decision Tree:                feature_importances_
    #5 Random Forest:                feature_importances_ 
    #6 SVM for regression:           coef_ (FOR LINEAR KERNEL ONLY!!!!):
    #7 Gradient Boosting Regression: feature_importances_
    #8 MLP:                          coefs_ (NOTICE THE s, it doesn't work)
    
    optimal_features_for_each_model = []

    print('____________________________________________________________________________________________')
    print('Summary of recursive feature elimination ')
    print('____________________________________________________________________________________________')
       
    #SVM only has the attribute coef_ for linear kernel, so in order to 
    #prevent errors it has not been considered for recursive feature 
    #elimination
    
    #MLP doesn't have coef_ attribute but coefs_ so it was supressed 
    #to prevent errors. 
    
    models_special = ['SVR','SVC','MLPRegressor','MLPClassifier']
    
    for idx in range(len(p)):
        
        if (p[idx].pipeline_name in models_special):
            #add all attributes in these cases.                                                                                        
            optimal_features_for_each_model.append(df_X.columns.values)                         
            print('------- features for %-30s are: %s'% (p[idx].pipeline_name,
                                                         df_X.columns.values))
        else:
            estimator_model = deepcopy(p[idx].best_estimator_model)
            extra_title_string = ('(%s)' % p[idx].pipeline_name)
            names_of_optimal_features  = recursive_feature_elimination_with_cross_validation(df_X,df_y,estimator_model,features_to_eliminate_per_step,k_folds,scoring,verbose,extra_title_string,split_type)
            optimal_features_for_each_model.append(names_of_optimal_features) 
            print('Optimal features for %-30s are: %s'% (p[idx].pipeline_name,
                                                    names_of_optimal_features))
    print('____________________________________________________________________________________________')
    print('\n')
    return deepcopy(optimal_features_for_each_model)


def recursive_feature_elimination_with_cross_validation(df_X,df_y,estimator_model,features_to_eliminate_per_step=1,k_folds=5,scoring='r2',verbose=True,extra_title_string='',split_type='simple'):
    r"""
    Recursive feature elimination with cross-validation.
    
    Parameters
    ----------
    df_X : pandas DataFrame
        Input data frame.
    df_y : pandas DataFrame
        Output data frame.
    estimator_model : ML estimator to test on input data.
    
    features_to_eliminate_per_step : int
        How many features should be eliminated in each round.
    k_folds : int
        Number of folds to use for the cross-validation.
    scoring : str
        Which performance metric will be used to assess the "importance" each feature in the model.
    verbose : bool
        Variable used to control if results are displayed (True) or not (False)
    extra_title_string : str
        Text added to "Cross validation score vs. Number of features selected" 
        in the figure title.
    
    Returns
    -------
    list
        List with the name of the optimal features.
    """
    #--------------------------------------------------------------------------
    #Get values from data frames.
    #--------------------------------------------------------------------------
    X=df_X.values
    y=df_y.values.ravel()
    #--------------------------------------------------------------------------
    
    rfecv = 0
    if split_type == 'simple':
        rfecv = RFECV(estimator=estimator_model, 
                      step=features_to_eliminate_per_step, 
                      cv=k_folds, scoring=scoring)
    elif split_type == 'stratified':
        rfecv = RFECV(estimator=estimator_model, 
                      step=features_to_eliminate_per_step, 
                      cv=model_selection.StratifiedKFold(k_folds), scoring=scoring)
    rfecv.fit(X, y)
    
    #--------------------------------------------------------------------------

    if (verbose==True):
        #print("Optimal number of features:\t %d out of %d"  % (rfecv.n_features_,len(df_X.columns.values)))
        
        #print("Input features: \t %s"  % df_X.columns.values)
        #print("Mask of selected features:\t %s"  % rfecv.support_)

        # Plot number of features VS. cross-validation scores
        plt.figure()
        plt.title('Cross validation score vs. Number of features selected %s' \
                  % extra_title_string)
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.show()
    #-------------------------------------------------------------------------------------
    names_of_optimal_features = []
    for idx in range(len(rfecv.support_)):
        if rfecv.support_[idx] == True:
            names_of_optimal_features.append(df_X.columns.values[idx])
    
    return deepcopy(names_of_optimal_features)


def pred_for_score(df_y, y_predict, performance_metric):
    r"""
    Use the corresponding prediction score according to the score parameter.
    
    Parameters
    ----------
    df_y : ndarray
        Ground truth values.
    y_predict : ndarray
        Predicted values.
    performance_metric : str
        Name for the score.
    
    Returns
    -------
    numeric
        The value of the selected performance metric.
    
    """
    if performance_metric == 'r2':
        return metrics.r2_score(df_y.ravel(), y_predict.ravel())
    
    elif performance_metric == 'neg_mean_squared_error':
        return metrics.mean_squared_error(df_y,y_predict)
    
    elif performance_metric == 'neg_log_loss':
        return metrics.log_loss(df_y,y_predict)
    # The scores f1, precision and recall can have the suffixes : macro,micro,
    # weighted and samples,  then it's necesarry to divide the name in 
    # two parts, the first part is the name of the score, that's why the name
    # is only checked to certain number, 2 for f1, 9 for precision, and 6 for
    # recall, the second part of the name is used as a parameter for the score
    # Example: for f1_weighted, the first part will be 'f1' and the second will
    # be 'weighted', is used as the paramater average
    elif performance_metric[0:2] == 'f1':
        if len(performance_metric) == 2: 
            return metrics.f1_score (df_y,y_predict)
        return metrics.f1_score(df_y,y_predict,average=performance_metric[3:])
    
    elif performance_metric[0:9] == 'precision':
        if len(performance_metric) == 9:
            return metrics.precision_score(df_y,y_predict)
        return metrics.precision_score(df_y,y_predict, average=performance_metric[10:])
    
    elif performance_metric[0:6] == 'recall':
        if len(performance_metric) == 6:
            return metrics.recall_score(df_y,y_predict)
        return metrics.recall_score(df_y,y_predict, average=performance_metric[7:])
    
    elif performance_metric == 'accuracy':
        return metrics.accuracy_score(df_y,y_predict)
    
    else:
        raise Exception('Performance metric %s is not available' % (performance_metric))
        
    
def compute_performance_metrics_for_all_prediction_models(p,
                                              optimal_features_for_each_model,
                                              df_X_test,df_y_test,scoring,
                                              model_type):
    """
    This function computes performance metrics for all models.
    
    Parameters
    ----------
    p : list 
        List of models (i.e: list of prediction_class objects).  
    optimal_features_for_each_model : list
        List of best features for each model.
    df_X_test : DataFrame
        Input dataframe for test set
    df_y_test : Dataframe
        Target dataframe for test set
    
    Returns
    -------
    p : list
        List of models
    """
    for idx in range(len(p)):
        optimal_features_for_current_model = deepcopy(
                optimal_features_for_each_model[idx])
        
        all_observations_in_test_set_of_selected_features = (
                df_X_test[optimal_features_for_current_model]).values                    
        #Compute predictions 
        y_predict = p[idx].best_pipeline.predict(
                all_observations_in_test_set_of_selected_features )  
        
        p[idx].performance_metric_value = pred_for_score(df_y_test.values.ravel(),y_predict,
                                     scoring)
        if model_type == 'classification':
            mat = metrics.confusion_matrix(df_y_test.values.ravel(),y_predict)
            rep = metrics.classification_report(df_y_test.values.ravel(),y_predict)
            p[idx].confusion_matrix = mat
            p[idx].classification_report = rep
    return p


def extract_best_pipeline_from_the_best_models(best_pipelines):
    """
    This function receives a list of objects of prediction_class that have 
    been trained and returns the best one. 
    
    Parameters
    ----------
    best_pipelines : list
        List of objects of prediction_class.
    
    Returns
    -------
    prediction_class object
        The best pipeline within the list of pipelines.
    """    
    best_model_pipeline = None
    score_name = best_pipelines[0].performance_metric_name
    # Value to decide if the score should be maximized or minimized
    comp_value = 1
    # If the score name ends with _error or _loss, then it should be
    # minimized. See https://scikit-learn.org/stable/modules/model_evaluation.html
    if score_name.endswith('_error') or score_name.endswith('_loss'):
        comp_value = -1

    best_score = -1*comp_value*np.inf

    for model_idx in range(len(best_pipelines)):
        #The best model is selected accordingly to the respective score
        if comp_value*best_pipelines[model_idx].performance_metric_value > comp_value*best_score:
            best_model_pipeline = deepcopy(best_pipelines[model_idx])
            best_score = best_model_pipeline.performance_metric_value
    return best_model_pipeline


def extract_best_pipelines_from_all_iterations(outputs_after_all_iterations):
    """
    This function takes the output of the function get_best_models and 
    checks all iterations and uses the best performing models.
    
    Parameters
    ----------
    outputs_after_all_iterations : list
        List by iterations of lists of objects of prediction_class.
    
    Returns
    -------
    best_pipelines : list
        List of objects of prediction_class
    """
    best_pipelines = []
    
    # We can select the first model for the first iteration becauses every
    # model has the same score name
    score_name = outputs_after_all_iterations[0][0].performance_metric_name
    # Value to decide if the score should be maximized or minimized
    comp_value = 1
    # If the score name ends with _error or _loss, then it should be
    # minimized. See https://scikit-learn.org/stable/modules/model_evaluation.html
    if score_name.endswith('_error') or score_name.endswith('_loss'):
        comp_value = -1
            
    for model_idx in range(len(outputs_after_all_iterations[0])):      
        best_score = -1*comp_value*np.inf
        best_model_pipeline = None
        for iter_idx in range(len(outputs_after_all_iterations)):
            actual_score = outputs_after_all_iterations[iter_idx][model_idx].performance_metric_value
            if actual_score*comp_value > comp_value*best_score:
                best_model_pipeline = deepcopy(outputs_after_all_iterations[iter_idx][model_idx])
                best_score = actual_score
        best_pipelines.append(deepcopy(best_model_pipeline))
        
    return best_pipelines


def compute_predictions_for_a_single_pipeline(p,df_X):
    """
    This function finds predictions for a single pipeline (it is assumed that 
    this is already the best model)

    Parameters
    ----------
    p : prediction_class object
    df_X: DataFrame
        Input dataframe with possible all the original attributes. 

    Returns
    -------
    ndarray
        Numpy array with output predictions.
    """
    #This is to check if there are optimal attributes of if all of the input 
    #attributes should be used.
    if len(p.names_of_optimal_features)>0:
        optimal_features_for_current_model = \
                                          deepcopy(p.names_of_optimal_features)
        dataset_with_best_features = \
                              (df_X[optimal_features_for_current_model]).values        
        
        #Compute predictions 
        y_predict = p.best_pipeline.predict(dataset_with_best_features)  
    else:  #Use all attributes for the prediction. 
        #Compute predictions
        y_predict = p.best_pipeline.predict(df_X.values)   

    return y_predict


def get_best_models(df_X,
                    df_y,
                    random_state                    = 42,
                    number_of_iterations            = 5,
                    compute_higher_order_features   = False,  
                    use_interaction_features_only   = True,   
                    degree_of_polynomial            = 2,      
                    global_percentage_for_test_size = 0.1,
                    local_percentage_for_test_size  = 0.1,
                    input_scaler                    = preprocessing.StandardScaler(),
                    k_folds                         = 5,
                    scoring                         = 'r2',
                    model_type                      = 'regression',
                    features_to_eliminate_per_step  = 1,
                    verbose_level                   = 0,                    
                    number_of_parallel_jobs         = -1,
                    parameters_file                 = "",
                    split_type                      = 'simple',
                    iid                             = False):
    """
    This function performs hyperparameter tuning, recursive feature 
    elimination, trains with best combination of both 
    (features and hyperparameters), and compute performance metrics on a 
    test set.
    
    Parameters
    ------------
    df_X : DataFrame
        Dataframe with input variables (rows are observations, 
        columns are features) 
    df_y : Dataframe
        Dataframe with output (or target) variable.
    random_state : int
        Random seed for the initial train test split.
    number_of_iterations : int
        Number of trials used to process the models with different splits of 
        data. 
    compute_higher_order_features : bool
        Set to False if you don't want to use high-order features.
    use_interaction_features_only : bool
        Set to False if you also want the whole polynomial. Set to True 
        to compute interaction features only.
    degree_of_polynomial : int
        Degree of the polynomial used to generate higher-order features.
    global_percentage_for_test_size : float
        Fraction of input examples devoted entirely for testing.
    local_percentage_for_test_size : float
        Local Fraction of input examples devoted entirely for testing 
        (the dataset will be split again inside the function 
        apply_machine_learning_pipeline).
    input_scaler : sklear.Scaler
        The options are: StandardScaler() or MinMaxScaler(), RobustScaler(), 
        Normalizer, etc...
    k_folds : int
        Number of folds in the cross validation scheme used for model 
        selection (i.e: Hyperparameter tuning).
    scoring : str
        Metric used to evaluate the fitness of the selected model for a given 
        set of hyperparameters.
    model_type : str
        Model's type to be fitted, 'regression' or 'classification'
    features_to_eliminate_per_step : int
        How many features to eliminate per step during the recursive feature 
        elimination process.
    verbose_level : int
        The higher this number the more verbose the output. If set to 0 it 
        doesn't display any intermediate processes, 10 shows everything.
    number_of_parallel_jobs : int 
        If set to 1: the grid search uses 1 core and it is useful for 
        debugging; is set to -1 the grid search uses all available cores.
    parameters_file : str
        Json with the models and parameters to be used
    split_type : str
        'simple' for random splittying, or 'stratified' to split
        according to the classes
    iid : bool
        If the data is iid (Independent and identically distributed)
        
    Returns
    -------
    list
        List of list of prediction class objects, one for each iteration
        in the process.
    """    
    feature_names=list(df_X.columns.values)
    check_score(scoring,model_type)
    #------------------------------------------------------------------------------------------------------
    #Higher order features: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
    #------------------------------------------------------------------------------------------------------
    
    if (compute_higher_order_features==True):
        #Note: #In some cases it’s not necessary to include higher powers of any single feature, 
        #but only the so-called interaction features that multiply together at most d distinct features. 
        #These can be gotten from PolynomialFeatures with the setting interaction_only=True.
        
        poly = preprocessing.PolynomialFeatures(degree=degree_of_polynomial,
                                                interaction_only=use_interaction_features_only)     
        x_poly = poly.fit_transform(df_X.values)
        target_feature_names = poly.get_feature_names(feature_names)
        
        print('____________________________________________________________________________________________')
        if (use_interaction_features_only==False):
            print('New features of order %d including all of them.'  % degree_of_polynomial)
        else:
            print('New features or order %d including the interaction between features only.'  % degree_of_polynomial)
        print('____________________________________________________________________________________________')
        print(target_feature_names)
        print('____________________________________________________________________________________________')
        print('\n\n')
    
        #Overwrite the original dataframe with all the new data.
        df_X = pd.DataFrame(x_poly, columns = target_feature_names)
        #print(df_X.describe()) #Quick summary of data.

    #------------------------------------------------------------------------------------------------------
    np.random.seed(random_state)                                             #Set the random seed at the begining of the process !!!!!!!!!
    
    outputs_after_all_iterations                         = []

    for num_iter in range(number_of_iterations):
        print('Iteration #%d out of %d' % (num_iter+1,number_of_iterations))       
        #------------------------------------------------------------------------------------------------------
        #Split the initial dataset, leaving a small part of it only for testing at the very end of this script!!!!
        #------------------------------------------------------------------------------------------------------
        test_rows = []
        df_X_train,df_X_test,df_y_train,df_y_test, test_rows = dataframe_split(
                df_X, df_y, global_percentage_for_test_size, split_type)
        
        #------------------------------------------------------------------------------------------------------
        #Call the machine learning pipeline
        #------------------------------------------------------------------------------------------------------
        optimal_features_for_each_model = []                                #List with optimal features for each model. 
        print('Phase 1: Hyperparameter tuning using all features.')
        p=[]
        p=apply_prediction_pipeline(df_X_train,
                                    df_y_train,
                                    optimal_features_for_each_model,  #Initially empty !!!.
                                    local_percentage_for_test_size,
                                    input_scaler,
                                    k_folds,
                                    scoring,
                                    model_type,
                                    split_type,
                                    number_of_parallel_jobs,
                                    verbose_level,
                                    parameters_file,
                                    iid)
        #------------------------------------------------------------------------------------------------------
        #Perform recursive feature elimination 
        #------------------------------------------------------------------------------------------------------
        verbose                         = False #True if you want to see an additional graph, False otherwise.
        print('Phase 2: Recursive feature elimination using best hyperparameters.')
        if features_to_eliminate_per_step == 0:
            print('Features to eliminate per step is zero, so this phase is not executed.')
            print('Phase 3: Extracting performance metrics for the test set.')
        
            p2 = compute_performance_metrics_for_all_prediction_models(deepcopy(p)
                     ,deepcopy(optimal_features_for_each_model),df_X_test,df_y_test
                     ,scoring,model_type)
            extra_title_string =' (GLOBAL test set)'
            print_results_for_tested_prediction_models(p2,extra_title_string)
            outputs_after_all_iterations.append(deepcopy(p2))
            continue
        else:
            optimal_features_for_each_model = \
               get_optimal_features_for_each_model(p,df_X_train,df_y_train,scoring,
                                    features_to_eliminate_per_step,k_folds,verbose)
    
        #------------------------------------------------------------------------------------------------------
        #Perform feature importance evaluation in models based on ensemble methods *******************
        #------------------------------------------------------------------------------------------------------
        #This is addtitional and optional...
        
        #print('Optional Phase: Importance feature selection for the Gradient Boosting Regressor.')
        #extra_title_string = '(Gradient Boosting Regressor)'
        #show_feature_importance(p[7].best_estimator_model ,df_X_train.columns,extra_title_string) #Pass the model and the names of input features in the model.
    
        #-------------------------------------------------------------------------------------
        #Perform again the grid search and hyperparameter tunning but only using the best features. 
        #-------------------------------------------------------------------------------------
        print('Phase 3: Hyperparamter tuning using only the optimal features \
              for each model.')
        p2=[]
        p2=apply_prediction_pipeline(df_X_train,
                                     df_y_train,
                                     optimal_features_for_each_model,  #Initially empty !!!.
                                     local_percentage_for_test_size,
                                     input_scaler,
                                     k_folds,
                                     scoring,
                                     model_type,
                                     split_type,
                                     number_of_parallel_jobs,
                                     verbose_level,
                                     parameters_file,
                                     iid)   
        
        #Preserve the names of the optimal features with in the regression_class
        for idx in range(len(p2)):
            p2[idx].names_of_optimal_features  = \
                                 deepcopy(optimal_features_for_each_model[idx])
            p2[idx].test_rows = test_rows
                
        #-------------------------------------------------------------------------------------
        #Get performance metrics on the unused test set.
        #-------------------------------------------------------------------------------------
        print('Phase 4: Extracting performance metrics for the test set.')
        
        p2 = compute_performance_metrics_for_all_prediction_models(deepcopy(p2)
                 ,deepcopy(optimal_features_for_each_model),df_X_test,df_y_test
                 ,scoring,model_type)
        extra_title_string =' (GLOBAL test set)'
        print_results_for_tested_prediction_models(p2,extra_title_string)
        outputs_after_all_iterations.append(deepcopy(p2))

    return outputs_after_all_iterations


def apply_prediction_pipeline(df_X,df_y,optimal_features_for_each_model=[],
                              test_size=0.1,input_scaler=preprocessing.StandardScaler(),
                              k_folds=5,scoring='', model_type = 'regression',
                              split_type = 'simple', 
                              number_of_parallel_jobs = -1,verbose_level=10,
                              parameters_file="",iid = False):
    
    """
    This function applies a machine learning pipeline to perform predict on 
    input x and output y.
    
    Parameters
    ------------
    df_X : DataFrame
        Dataframe with input data (columns are attributes and rows 
        are observations).
    df_y : DataFrame
        Data frame with output data (columns are outputs and rows 
        are observations).
    test_size : numeric
        Fraction of observations devoted for testing, the rest is 
        used for training in a cross-validation scheme.
    input_scaler : sklear.Scaler
        How do you want to scale your inputs: e.g: StandardScaler() or 
        MinMaxScaler(), RobustScaler(), Normalizer()        
    k_folds : int
        Number of folds used for cross validation.
    scoring : str
        Metric used to evaluate performance.
    mode_type : str
        It can be either 'regression' or 'classification'.
    split_type : str
        It can be either 'simple' or 'stratified'.
    number_of_parallel_jobs : int
        If set to 1 the grid search uses 1 core, this is useful for debugging; 
        if set to -1 the grid search uses all cores available.
    verbose_level : int
        This is an integer variable the larger it is, the more information you 
        get during the grid search process.
    parameters_file : str
        Json with the models and parameters to be used
        
    Returns
    -------
    list
        List with the prediction_class object with the tuned hyperparameters.
    """
    
    #Check if the score is correctly assigned to the model type
    check_score(scoring,model_type)
    #list of pipelines
    p = []
    # Create the pipelines according to the model type
    #json_file = '/Users/yoksil/Dropbox/Universidad/2019-1/PI1/codes/refactoringML/main/parameters3.json'
    
    with open(parameters_file) as f:
        data = json.load(f)
    p = apply_prediction_pipeline_aux(model_type,input_scaler,k_folds, scoring, 
                                   number_of_parallel_jobs, verbose_level,
                                   split_type=split_type,data=data,
                                   iid_param = iid)
                                    
    
    #Split input data (this time we are going to use the data frame and not 
    # the numpy array for convenience)
    df_X_train,df_X_test,df_y_train,df_y_test,_ = dataframe_split(df_X,df_y,
                                                                test_size,
                                                                split_type)
    
    #-------------------------------------------------------------------------------------
    #Iterate over each pipeline (apply scaling, grid search, and training)
    #-------------------------------------------------------------------------------------
    
    #Note:- The estimators of a pipeline are stored as a list in the steps 
    #attribute, for instance: pipe.steps[0]
    #       and as a dict in named_steps: pipe.named_steps['Scaler']
    #     - Parameters of the estimators in the pipeline can be accessed using 
    # the <estimator>__<parameter> syntax: pipe.set_params(Estimator_SVR__C=10)
    
    #If the user wants to use all features for all models, then:
    if (len(optimal_features_for_each_model)==0):  
        for idx in range(len(p)):
            optimal_features_for_each_model.append(df_X.columns.values)
    
    for idx in range(len(p)):
        print('Fitting %s.' % p[idx].pipeline_name)
        
        optimal_features_for_current_model = deepcopy(optimal_features_for_each_model[idx])
        
        all_observations_in_training_set_of_selected_features = (
                df_X_train[optimal_features_for_current_model]).values        
        
        all_observations_in_test_set_of_selected_features = (
                df_X_test[optimal_features_for_current_model]).values                    

        p[idx].names_of_optimal_features = deepcopy(optimal_features_for_current_model)
                                   
        p[idx].best_pipeline.fit(all_observations_in_training_set_of_selected_features, 
                                 df_y_train.values.ravel())
        
        if  p[idx].grid_search_flag==True:
            #Save best model (notice that this doesn't include the scaler for instance
            step_name, p[idx].best_estimator_model = \
                       deepcopy(p[idx].best_pipeline.best_estimator_.steps[-1])
            p[idx].tuned_hyperparameters           = deepcopy(p[idx].best_pipeline.best_params_)               #Save the best tuned hyperparameters.
            p[idx].all_hyperparameters             = deepcopy(p[idx].best_estimator_model.get_params())        #Save all the hyperparameters (this is a super set of the previous one)   
            p[idx].best_pipeline                   = deepcopy(p[idx].best_pipeline.best_estimator_)            #Leave this update at the end of this block, in other words, don't move it. 
        else:   #In this case the existing pipeline is always the best pipeline as there is no grid search. 
            #p[idx].best_pipeline
            p[idx].best_estimator_model = deepcopy(p[idx].best_pipeline.steps[-1][-1])  #Last step (row), and process (column) of the pipeline.
            p[idx].all_hyperparameters  = deepcopy(p[idx].best_estimator_model.get_params())
           
        y_predict    = p[idx].best_pipeline.predict(all_observations_in_test_set_of_selected_features)  #Compute predictions 
        p[idx].performance_metric_value = pred_for_score(df_y_test.values.ravel(),y_predict,scoring)
    #-------------------------------------------------------------------------------------
    #Display best models and the corresponding performance metrics. 
    #------------------------------------------------------------------------------------- 
    title_string=' (LOCAL test set)'
    print_results_for_tested_prediction_models(p,title_string)  
    #------------------------------------------------------------------------------------- 
    
    return deepcopy(p)  #The output is returned in this object.

def get_estimator(name):
    """
    Return the corresponding estimator.
    
    Parameters
    ----------
    name : str
        Name of the estimator
    
    Returns
    -------
    Estimator
        The corresponding estimator.
    """
    predictors = ['LinearRegression','Ridge','Lasso','BayesianRidge',
                  'DecisionTreeRegressor','RandomForestRegressor','SVR',
                  'GradientBoostingRegressor','MLPRegressor',
                  'RandomForestClassifier','ExtraTreesClassifier','SVC',
                  'MLPClassifier', 'MultinomialNB']
    
    if name not in predictors:
        raise Exception('Estimator %s is not available' % (name))
    
    name = name + '()'
    return eval(name)
    
        
def apply_prediction_pipeline_aux(model_type,input_scaler=preprocessing.StandardScaler(), 
                                  k_folds=5,scoring='r2', 
                                  number_of_parallel_jobs = -1,verbose_level=10
                                  ,data={},split_type='simple',
                                  iid_param = False):
    """
    Auxiliar functions to parse the json file and create the pipelines with 
    the corresponding parameters
    
    Parameters
    ------------
    input_scaler : sklearn.Scaler
        How do you want to scale your inputs: e.g: StandardScaler() or 
        MinMaxScaler(), RobustScaler(), Normalizer()
    k_folds : int
        Number of folds used for cross validation.
    scoring : str
        Metric used to evaluate performance.
    number_of_parallel_jobs : int
        If set to 1 the grid search uses 1 core, this is useful for debugging; 
        if set to -1 the grid search uses all cores available.
    verbose_level : int
        This is an integer variable the larger it is, the more information you 
        get during the grid search process.
    data : dict
        Json file as dictionary with the models and parameters to be used
    
    Returns
    -------
    list
        List of prediction_class object
    """
    #Get the list of models
    models = data['models']
    pipes = []
    
    for m in models:
        #Get the name of models
        model_name = m['name']
        check_model_type(model_name, model_type)
        grid_search = True
        #If the parameter dictionary is empty then we can't apply grid search
        if 'parameters' not in m.keys():
            grid_search = False
        if 'scaler' in m.keys():
            input_scaler = eval(m['scaler']+"()")
        
        estimator_name = 'Estimator_' + model_name
        #Create a pipelines with the scaler and estimator
        pipeline_pred = Pipeline(steps=[('Scaler_' + model_name, input_scaler ),
                                         (estimator_name, get_estimator(model_name))])
        
        if grid_search:
            param = m['parameters']
            
            #Change the name of the parameters according with the estimator
            #Every parameter now will have the form: 'estimator__parameter',the
            #double under score is something required by the sklearn
            for p in param:
                dict_k = list(p.keys())
                for x in dict_k:
                    #Tuples in hidden layer sizes and booleans in fit_intercept
                    #are not valid as json parameters, then it's necessary to
                    #read as string and then evaluate it
                    if x == 'hidden_layer_sizes' or x == 'fit_intercept':
                        p[x] = [eval(i) for i in p[x]]
                    p[estimator_name + "__" + x] = p.pop(x)
            
            #Create the corresponding Grid Search
            #Use the proper split type
            if split_type == 'simple':
                estimator_pred = model_selection.GridSearchCV(
                        estimator=pipeline_pred, param_grid=param, scoring=scoring,
                        cv=k_folds, refit=True, n_jobs=number_of_parallel_jobs,
                        verbose=verbose_level, iid=iid_param)
            elif split_type == 'stratified':
                estimator_pred = model_selection.GridSearchCV(
                        estimator=pipeline_pred, param_grid=param, scoring=scoring,
                        cv=model_selection.StratifiedKFold(k_folds), refit=True,
                        n_jobs=number_of_parallel_jobs, verbose=verbose_level,
                        iid=iid_param)

            pi = prediction_class(model_name, best_pipeline=estimator_pred,
                              grid_search_flag=True,performance_metric_name=scoring)
        else:
            pi = prediction_class(model_name, best_pipeline=pipeline_pred,
                              grid_search_flag=False,performance_metric_name=scoring)
        pipes.append(pi)
    
    return pipes


def show_performance_metrics(outputs_after_all_iterations, 
                             name_of_x                      = 'MSE',
                             bandwidth_to_use               = 'Scott',
                             kernel                         = 'gaussian',
                             num_points_to_generate_in_kde_graph = 400,
                             share_x_axis_among_all_charts  = True,
                             title_string              = 'Case study XYZ',
                             flag_show_plot_in_different_rows = False,
                             linewidth                      = 2,
                             fontsize                       = 12,
                             list_with_spacing_options      = [0.90, 0.10, 0.10, 0.90, 0.2, 0.2], 
                             figsize                        = (10, 15),
                             flag_save_figure               = True,
                             output_path                    = '/home/',
                             filename_without_extension     = 'figure_with_probability_density_functions_of_performance_metrics_after_autotuning',
                             extension                      = '.pdf'):
    """
    Parameters
    ---------- 
    output_after_all_iterations : list    
    name_of_x : str
        Name of x-axis that corresponds to the metric that you are evaluating. 
        For instance 'R²' or 'MSE' or 'F1'.
    bandwidth_to_use : str
        This specifies the bandwidth to use in the kernel density estimation 
        process. Supported options include  'Scott', 'Silverman'.
    kernel : str
        Kernel to use in the r Kernel Density Estimation. 
        The options are: 'gaussian, 'tophat', 'epanechnikov', 'exponential', 
        'linear', 'cosine'.     
    num_points_to_generate_in_kde_graph : int
        How many points are going go to be used to generate the KDE contour.  
    share_x_axis_among_all_charts : bool
        If set to True, the same x-axis limits are used for ALL models, 
        otherwise each model has its own x-axis limits
    title_string : str
        Title for the case study of the figure. 
    flag_show_plot_in_different_rows : bool
        If True the plot is created with one row per KDE, otherwise all the 
        KDEs are shown in 1 row.
    linewidth : int
        Line width for the KDE plot
    fontsize : int
        Font size of the figure. 
    list_with_spacing_options : list
        List with floating-point values to control the spacing within the 
        figure using matplotlib convention [top, bottom, left, right, hspace, wspace].
    figsize : tuple
        Overall figure size. For instance (10, 15).
    flag_save_figure : bool
        If set to True, the function saves the figure in the HDD. 
    output_path : str
        String that points to the output path for saving the resulting image. 
        For instance '/home/'
    filename_without_extension : str
        String of the filename to use for saving the figure. 
        For instance: 'figure_with_probability_density_functions_of_performance_metrics_after_autotuning'
    extension : str
        Image extension. For instance '.pdf' or '.png'
    """
    #Extract number of trials and number of models, create a dataframe, etc...
    num_trials       = len(outputs_after_all_iterations)
    num_models       = len(outputs_after_all_iterations[0])
    names_of_models  = []

    # We can pick the score name of any element because it is the same for
    # every element
    score_name = outputs_after_all_iterations[0][0].performance_metric_name
    #Initialize matrices. 
    x_matrix = np.zeros(shape=(num_trials,num_models))
    
    #For the first trial, extract the model names available...
    for j in range(num_models):
        names_of_models.append(deepcopy(outputs_after_all_iterations[0][j].pipeline_name))
    
    #For all trials, and for all models.....
    for i in range(num_trials):
        for j in range(num_models):
            x_matrix[i][j] = outputs_after_all_iterations[i][j].performance_metric_value

    pd_x = pd.DataFrame(x_matrix,  columns=names_of_models)

    # Get the mean score value for each model
    list_of_tuple_mean_score_name = []
    for col in list(pd_x.columns):
        values = np.array(pd_x[col])
        mean_score = np.mean(values)
        list_of_tuple_mean_score_name.append((mean_score, col))
    
    # Order the list according to the mean score value. This ordering is
    # ascending
    list_of_tuple_mean_score_name = sorted(list_of_tuple_mean_score_name)
    
    # If the score name ends with _score, then it means that the greater the
    # better, so we must reverse the list
    if score_name.endswith('_score') is True:
        list_of_tuple_mean_score_name.reverse()

    new_column_name_order = []
    for tup in list_of_tuple_mean_score_name:
        new_column_name_order.append(tup[1])
        
    pd_x = pd_x[new_column_name_order]

    output_path = ''
    compute_and_display_the_KDE_from_a_dataframe(pd_x,
                             name_of_x                      = name_of_x,
                             bandwidth_to_use               = bandwidth_to_use,
                             kernel                         = kernel,
                             num_points_to_generate_in_kde_graph = num_points_to_generate_in_kde_graph,  
                             share_x_axis_among_all_charts  = share_x_axis_among_all_charts,
                             title_string                   = title_string,
                             flag_show_plot_in_different_rows = flag_show_plot_in_different_rows, 
                             linewidth                      = linewidth,
                             fontsize                       = fontsize,
                             list_with_spacing_options      = list_with_spacing_options, 
                             figsize                        = figsize,
                             flag_save_figure               = flag_save_figure,
                             output_path                    = output_path,
                             filename_without_extension     = filename_without_extension,
                             extension                      = extension)

def compute_and_display_the_KDE_from_a_dataframe(pd_x,
                                                 name_of_x                      = 'MSE',
                                                 bandwidth_to_use               = 'Scott',
                                                 kernel                         = 'gaussian',
                                                 num_points_to_generate_in_kde_graph = 400,  
                                                 share_x_axis_among_all_charts  = True,
                                                 title_string              = 'Case study XYZ',
                                                 flag_show_plot_in_different_rows = False, 
                                                 linewidth                      = 2,
                                                 fontsize                       = 12,
                                                 list_with_spacing_options      = [0.90, 0.10, 0.10, 0.90, 0.2, 0.2], 
                                                 figsize                        = (10, 15),
                                                 flag_save_figure               = True,
                                                 output_path                    = '/home/',
                                                 filename_without_extension     = 'figure_with_probability_density_functions_of_performance_metrics_after_autotuning',
                                                 extension                      = '.pdf'):

    """
    This function shows the performance metric of a set of models trained with the autotuning program. 
    
    Parameters
    ---------- 
    pd_x : object
        Pandas dataframe where the rows are the number trials (i.e: observations), and the columns are the number of models.
    filename_for_input_pickle_file : string 
        Complete path and filename with extension to the pickle file that was used to store the autotuning results.
        This object includes the variable outputs_after_all_iterations creating by the autotuning. 
    name_of_x : string
        Name of x-axis that corresponds to the metric that you are evaluating. For instance 'R²' or 'MSE' or 'F1'.
    bandwidth_to_use : string
        This specifies the bandwidth to use in the kernel density estimation process. Supported options include  'Scott', 'Silverman'.
    kernel : string
        Kernel to use in the r Kernel Density Estimation. The options are: 'gaussian, 'tophat','epanechnikov', 'exponential','linear','cosine'.     
    num_points_to_generate_in_kde_graph : int
        How many points are going go to be used to generate the KDE contour.
    share_x_axis_among_all_charts : bool
        If set to True, the same x-axis limits are used for ALL models, otherwise each model has its own x-axis limits
    title_string : string
        Title for the case study of the figure. 
    flag_show_plot_in_different_rows : bool
        If True the plot is created with one row per KDE, otherwise all the KDEs are shown in 1 row.
    linewidth : int
        Line width for the KDE plot
    fontsize : int
        Font size of the figure.
    list_with_spacing_options : list
        List with floating-point values to control the spacing within the figure using matplotlib convention [top, bottom, left, right, hspace, wspace].
    figsize : tuple
        Overall figure size. For instance (10, 15).
    flag_save_figure : bool
        If set to True, the function saves the figure in the HDD. 
    output_path : string
        String that points to the output path for saving the resulting image. For instance '/home/'
    filename_without_extension : string
        String of the filename to use for saving the figure. For instance: 'figure_with_probability_density_functions_of_performance_metrics_after_autotuning'
    extension : string
        Image extension. For instance '.pdf' or '.png'

    Returns
    -------
    None
    
    Examples
    -------- 
    .. code-block:: Python
    
        N=100
        var1 = list(1*np.random.randn(N) + 1)
        var2 = list(5*np.random.randn(N) -1 )
        list_of_tuples = list(zip(var1, var2)) # get the list of tuples from two lists and merge them by using zip().  
        columns = ['var1','var2']
        pd_x=pd.DataFrame(data=list_of_tuples,columns=columns)
        name_of_x    = 'Error of measurement'
        title_string = 'Experiment 1'
        flag_show_plot_in_different_rows = False 
        compute_and_display_the_KDE_from_a_dataframe(pd_x                                = pd_x,
                                                     name_of_x                           = name_of_x,
                                                     bandwidth_to_use                    = 'std', # #'Scott' #'Binwidth' #, 'Silverman'.
                                                     kernel                              = 'gaussian',
                                                     num_points_to_generate_in_kde_graph = 400,  
                                                     share_x_axis_among_all_charts       = True,
                                                     title_string                        = title_string,
                                                     flag_show_plot_in_different_rows    = flag_show_plot_in_different_rows, 
                                                     linewidth                           = 2,
                                                     fontsize                            = 12,
                                                     list_with_spacing_options           = [0.90, 0.10, 0.10, 0.90, 0.2, 0.2], 
                                                     figsize                             = (10, 5),
                                                     flag_save_figure                    = True,
                                                     output_path                         = '/home/alejandro/',
                                                     filename_without_extension          = 'figure_with_probability_density_functions',
                                                     extension                           = '.pdf')
    """
    #print(pd_x.describe()) #Quick summary of data.
    #print(pd_x.shape)       #Rows and columns of the dataframe
    
    #Extract number of trials and number of models, create a dataframe, etc...
    num_trials       = pd_x.shape[0]  
    num_models       = pd_x.shape[1]  

    #Extract minumum and maximum value for the current performance statistic for all models and trials.
    min_x = pd_x.values.min()
    max_x = pd_x.values.max()            #Note that in the case of of R² the maximum theoretical value is 1.
        
    if min_x==max_x:
        print('The minimum value and the maximum value for %s is %0.2f. Therefore there is no histogram to show.' % (name_of_x,min_x))
        return
    
    #Variables for histograms and kernel density estimation 
    #Note: We will use the Freedman-Diaconis rule to estimate the bin size for the histogram
    #"See: https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
    array_with_recommended_bin_sizes_for_x = np.zeros(num_models)    
    for idx in range(num_models):
        array_with_recommended_bin_sizes_for_x[idx] = (2* iqr(pd_x[pd_x.columns[idx]].values))/(num_trials**(1/3))
    recommended_bin_size_x = np.min(array_with_recommended_bin_sizes_for_x)  #Select the minumum bin size 
    
    if recommended_bin_size_x==0:
        print('An error has been found when computing the histogram of %s because the recommende bin size is 0.' % name_of_x)
        return
    
    #Aux variables
    num_bins_x             = np.ceil((max_x-min_x)/recommended_bin_size_x)  #Compute the number of bins required to cover.
    bins_for_x             = np.linspace(min_x, max_x, num=num_bins_x)      #Bins for histogram.
    bin_size_x             = bins_for_x[1]-bins_for_x[0]                    #This is the final bin size that will be used !!!!
    dimension_of_data      = 1.

    #List to simplify the plot me(x_matrix,  columns=names_of_models)aking.
    list_with_maximum_height_of_histogram_divided_by_num_trials = []
    #list_with_maximum_values_of_the_x_distribution_per_model    = []
    list_of_histogram_values_divided_by_num_trials =[]
    list_of_x_bin_edges = []
    
    #Compute some aux variables.
    for idx in range(num_models):
        histogram_values, x_bin_edges          = np.histogram(pd_x.iloc[:,idx].values.ravel(), bins=bins_for_x, range=(min_x,max_x))
        histogram_values_divided_by_num_trials = histogram_values/num_trials
        
        list_with_maximum_height_of_histogram_divided_by_num_trials.append(np.max(histogram_values_divided_by_num_trials))
        #list_with_maximum_values_of_the_x_distribution_per_model.append(np.max(histogram_values_divided_by_num_trials)/(bin_size_x*np.sum(histogram_values_divided_by_num_trials)))       
        list_of_histogram_values_divided_by_num_trials.append(histogram_values_divided_by_num_trials)
        list_of_x_bin_edges.append(x_bin_edges)
    num_bars = len(x_bin_edges)-1        
    bar_centers_in_x=np.zeros(num_bars)
    
    #Compute the bar_centers in case of ploting a histogram_values_divided_by_num_trials with plt.bars
    for j in range(num_bars):
        bar_centers_in_x[j] = (x_bin_edges[j]+x_bin_edges[j+1])/2.0
            
    #upper_limit_of_x_distributions_for_all_models  = np.max(list_with_maximum_values_of_the_x_distribution_per_model)  #Maximum values of normalized histogram.
    #max_height_of_histograms_divided_by_num_trials =np.max(list_with_maximum_height_of_histogram_divided_by_num_trials)

    #Compute data for the KDE.
    list_with_input_data_for_kde_function  = []
    list_with_probability_density_function = []
    list_with_bandwidth_x                  = []
    list_with_max_ylim_for_visualization_per_model = []
    for idx in range(num_models):
        #With matplotlib
        data_for_current_histogram = pd_x.iloc[:,idx].values.ravel()
        
        if share_x_axis_among_all_charts == True:
            min_value_for_current_model = min_x
            max_value_for_current_model = max_x
        else:
            min_value_for_current_model = np.min(data_for_current_histogram)
            max_value_for_current_model = np.max(data_for_current_histogram)
            
        #Now find the bandwidth for the kernel density estimation based on the histogram.
        bandwidth_x = []
        if bandwidth_to_use=='Scott':
            bandwidth_x     = num_trials**(-1./(dimension_of_data+4.0))                                     #Scott’s Rule:   
        elif bandwidth_to_use=='Silverman':
            bandwidth_x     = (num_trials * (dimension_of_data + 2) / 4.)**(-1. / (dimension_of_data + 4))  #Silverman’s Rule: 
        elif bandwidth_to_use=='std':
            bandwidth_x = np.std(data_for_current_histogram)
        else:    
            bandwidth_x = bandwidth_to_use   
                                                  
        input_data_for_kde_function = np.linspace(min_value_for_current_model, max_value_for_current_model, num_points_to_generate_in_kde_graph)[:, np.newaxis]
        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth_x).fit(data_for_current_histogram.reshape(-1,1))       
        log_density = kde.score_samples(input_data_for_kde_function)
        probability_density_function = np.exp(log_density)
        
        list_with_input_data_for_kde_function.append(input_data_for_kde_function)
        list_with_probability_density_function.append(probability_density_function)
        list_with_bandwidth_x.append(bandwidth_x)
        list_with_max_ylim_for_visualization_per_model.append(np.max(list_with_probability_density_function))            
        
    max_ylim_for_visualization = np.max(list_with_max_ylim_for_visualization_per_model)

    # =============================================================================
    #     Create graph depending on the visualization pattern
    # =============================================================================
    

    if flag_show_plot_in_different_rows == True:
        #Create figure and define subplots
        fig, axs = plt.subplots(nrows=num_models, ncols=1, figsize=figsize)
    
        #Recover spacing options from arguments.    
        top    = list_with_spacing_options[0]
        bottom = list_with_spacing_options[1]
        left   = list_with_spacing_options[2]
        right  = list_with_spacing_options[3] 
        hspace = list_with_spacing_options[4] 
        wspace = list_with_spacing_options[5]
        plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace)
    
        #Suptitle and title
        #fig.suptitle(title_string, fontsize=fontsize)
    
        #For each model:   
        for idx in range(num_models):
            
            if num_models>1:           #If there is more than 1 model
                current_ax = axs[idx]
            else:                      #If there is just 1 model (notice that the indexing causes troubles.)
                current_ax = axs
            
            if idx==0:
                current_ax.set_title(title_string, fontsize=fontsize)
            
            color_string = 'C'+str(idx)  #This creates a sequence of colors in matplotlib.
    
            #With matplotlib
            data_for_current_histogram = pd_x.iloc[:,idx].values.ravel()
            
            if share_x_axis_among_all_charts == True:
                min_value_for_current_model = min_x
                max_value_for_current_model = max_x
            else:
                min_value_for_current_model = np.min(data_for_current_histogram)
                max_value_for_current_model = np.max(data_for_current_histogram)
            
            #Plot Histogram
            #current_ax.hist(data_for_current_histogram, bins=bins_for_x, facecolor='black')
    
            #Plot Histogram values divided by num trials
            #current_ax.bar(bar_centers_in_x,list_of_histogram_values_divided_by_num_trials[idx], width=bin_size_x, color=color_string)
            
            #Plot Gaussian KDE
            current_ax.plot(list_with_input_data_for_kde_function[idx][:, 0], list_with_probability_density_function[idx], linestyle='-', color=color_string, linewidth=linewidth)
    
            xlabel_string = name_of_x # '{} value'.format(name_of_x)
            #ylabel_string = '{}'.format(names_of_models[idx])
            ylabel_string = 'PDF'
            if idx==num_models-1:
                current_ax.set_xlabel(xlabel_string, fontsize=fontsize)  #Show only one ylabel at the bottom.
            current_ax.set_ylabel(ylabel = ylabel_string, rotation='vertical', ha='right', fontsize=fontsize)
            
            legend_string = pd_x.columns[idx]+'\nKernel = {}.\nBandwidth = {:0.2e}.'.format(kernel, list_with_bandwidth_x[idx])
            current_ax.legend([legend_string], loc='upper right')
            #current_ax.grid()        
            
            current_ax.set_xlim([min_value_for_current_model,max_value_for_current_model])
            #current_ax.set_xticks(???, minor=False)
            #current_ax.set_ylim([0,max_height_of_histograms_divided_by_num_trials])
            current_ax.set_ylim([0,max_ylim_for_visualization])
    else:
        #Create figure and define subplots
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    
        #Recover spacing options from arguments.    
        top    = list_with_spacing_options[0]
        bottom = list_with_spacing_options[1]
        left   = list_with_spacing_options[2]
        right  = list_with_spacing_options[3] 
        hspace = list_with_spacing_options[4] 
        wspace = list_with_spacing_options[5]
        plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace)
    
        #Suptitle and title
        #fig.suptitle(title_string, fontsize=fontsize)
    
        #For each model:   
        list_of_legends = []        
        for idx in range(num_models):
            
            current_ax = axs
            
            if idx==0:
                current_ax.set_title(title_string, fontsize=fontsize)
                xlabel_string = name_of_x # '{} value'.format(name_of_x)
            
                #ylabel_string = '{}'.format(names_of_models[idx])
                ylabel_string = 'PDF'
                
                current_ax.set_xlabel(xlabel_string, fontsize=fontsize)  #Show only one ylabel at the bottom.
                current_ax.set_ylabel(ylabel = ylabel_string, rotation='vertical', ha='right', fontsize=fontsize)
            
                current_ax.set_xlim([min_value_for_current_model,max_value_for_current_model])
                #current_ax.set_xticks(???, minor=False)
                #current_ax.set_ylim([0,max_height_of_histograms_divided_by_num_trials])
                current_ax.set_ylim([0,max_ylim_for_visualization])
                
                plt.tick_params(labelsize=int(np.round(0.9*fontsize)))
            
            color_string = 'C'+str(idx)  #This creates a sequence of colors in matplotlib.
    
            #With matplotlib
            data_for_current_histogram = pd_x.iloc[:,idx].values.ravel()
            
            min_value_for_current_model = min_x
            max_value_for_current_model = max_x
                
            #Plot Histogram
            #current_ax.hist(data_for_current_histogram, bins=bins_for_x, facecolor='black')
    
            #Plot Histogram values divided by num trials
            #current_ax.bar(bar_centers_in_x,list_of_histogram_values_divided_by_num_trials[idx], width=bin_size_x, color=color_string)
            
            #Plot Gaussian KDE
            current_ax.plot(list_with_input_data_for_kde_function[idx][:, 0], list_with_probability_density_function[idx], linestyle='-', color=color_string, linewidth=linewidth)
    
            legend_string = pd_x.columns[idx]+'. Kernel = {}. Bandwidth = {:0.2e}.'.format(kernel, list_with_bandwidth_x[idx])
            
            list_of_legends.append(legend_string)
            
        current_ax.legend(list_of_legends, loc='upper right',fontsize=int(np.round(0.9*fontsize)))
        #current_ax.grid()        
        
    #Save figure.
    if flag_save_figure==True:
        fig.savefig(fname = output_path+filename_without_extension+extension, bbox_inches='tight')
    plt.show(block=False)