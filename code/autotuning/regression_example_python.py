#!/usr/bin/python3
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------------------
#Libraries
#------------------------------------------------------------------------------------------------------
import sys
import os
import pandas as pd
import numpy as np
import time
from sklearn import preprocessing

import autotuning
#------------------------------------------------------------------------------------------------------
#Main program
#------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # Create a random dataset with 4 variables and 100 observations. It is
    # important that the dataframe has a label for every variable as the
    # process selecets the best features for the model, an identifier is needed
    # for each variable.
    x_array = np.random.rand(100,4)
    
    # We generate random coefficients that will be used to generate the output
    coeffs = np.random.rand(1,4)
    # Generate the output and add some noise
    noise = np.random.rand(100,1) * 0.45
    y_array = np.sum(x_array*coeffs, axis=1).reshape(-1,1) + noise

    df_X = pd.DataFrame(x_array)
    df_X.columns = ['var0', 'var1', 'var2', 'var3']

    df_y= pd.DataFrame(y_array)
    df_y.columns = ['output']

    # You have to past the path and filename to the json file
    path_and_filename_to_json_file = 'regression_example_parameters.json'
    tic = time.time()
    outputs_after_all_iterations = autotuning.get_best_models(
            df_X,
            df_y,
            random_state                    = 42,
            number_of_iterations            = 20,
            compute_higher_order_features   = False,
            use_interaction_features_only   = True,
            degree_of_polynomial            = 2,      
            global_percentage_for_test_size = 0.1,
            local_percentage_for_test_size  = 0.1,
            input_scaler                    = preprocessing.StandardScaler(),
            k_folds                         = 10,
            scoring                         = 'r2',
            model_type                      = 'regression',
            features_to_eliminate_per_step  = 1,
            verbose_level                   = 0,                    
            number_of_parallel_jobs         = -1,
            parameters_file                 = path_and_filename_to_json_file,
            split_type                      = 'simple',
            iid                             = False)
    toc = time.time()
    elapsed_time = toc-tic
    print('The whole process took %0.2f seconds = %0.2f minutes = %0.2f hours'\
          % (elapsed_time,elapsed_time/60,elapsed_time/(60*60)))

    # Get the best result for each model
    best_pipelines = autotuning.extract_best_pipelines_from_all_iterations(outputs_after_all_iterations)
    # Get the best model from the best results
    best_pipeline  = autotuning.extract_best_pipeline_from_the_best_models(best_pipelines)
    
    for pipeline in best_pipelines:
        # Each of this pipeline is a prediction class object, so that we can access
        # its attributes
        print()
        print("Pipeline name : {}".format(pipeline.pipeline_name))
        print("The {} value for this model was: {}.".format(pipeline.performance_metric_name,
              pipeline.performance_metric_value))
        print()
        print("Best hyperparameters:")
        print(pipeline.tuned_hyperparameters)
        print()
        print("Optimal features:")
        print(pipeline.names_of_optimal_features)
        print()
        print("--------------------------------------------------------------")
    
    autotuning.show_performance_metrics(outputs_after_all_iterations,
                                    name_of_x = 'R2',
                                    title_string = 'Autotuning example with regression',
                                    share_x_axis_among_all_charts = True,
                                    flag_show_plot_in_different_rows = True,
                                    flag_save_figure = False)
