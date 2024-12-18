import os
import time
import datetime
import joblib

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import shapiro

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance


from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    explained_variance_score,

    accuracy_score,
    precision_score,
    recall_score,
    f1_score,

    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

from mealpy import FloatVar, StringVar, IntegerVar, BoolVar, MixedSetVar, Problem, ACOR, GA, PSO, SA

# Data Preprocessing

"""
Variables needed:
- None

Returns:
- Best model, saved in pkl
- result_df (pd.DataFrame; containing optimizer name, optimizers [obj], best_ml_model for each optimizer, eval_metrics, time taken)
- best_ml_model (obj)
- data_dict (dict; data dictionary based on task type. reg & clf = "X_train":[], ... "y_test":[] and cls = "X":[])
"""

# 1 Create Data dictionary
# 2 Get model and necessary variables
# 3 Get Evaluation metrics
# - Set minmax value
# 4 Set hyperparameter bounds
# - set epoch
# - set population size
# - define optimizers
# - define lists for evaluation
# - iterate through each optimizer
# 6 Dependecies handling
# 6 Define problem class
# - start time
# - Start optimization (optimizer.solve(problem))
# - end time
# - get best parameters
# - decode best parameters
# 7 Re-fit and re-predict using best parameters
# 7.1 Decode best paras
# 7.2 Re-fit and re-predict
# - append evaluation results into the lists defined above
# - print evaluation results
# - create result dataframe
# - set ascending value
# - assign best model into a variable, sorted by reference metric (followed by the ascending value)
# - save best model

# 1 Data Dictionary
def create_data_dict(data, task_type):

    if task_type in ('regression', 'classification'):
        data = {
            "X_train": data[0],
            "X_test": data[1],
            "y_train": data[2],
            "y_test": data[3], 
        }
        
    elif task_type == 'clustering':
        data = {"X" : data}

    return data

# 2 Get model and necessary variables
def model_and_variables(data, task_type):
    model = None
    reference_metric = None
    n_obsv = None
    n_predictors = None
    n_classes = None
    is_multioutput = None

    if task_type == 'regression':
        model = RandomForestRegressor
        reference_metric = "Mean Squared Error (MSE)"

        n_obsv = len(data["y_test"])  # Number of observations
        n_predictors = data["X_test"].shape[1]  # Number of predictors (features)

    elif task_type == 'classification':
        n_classes = len(np.unique(data["y_train"])) if data["y_train"] is not None else None
        is_multioutput = len(data["y_train"].shape) > 1 and data["y_train"].shape[1] > 1 if data["y_train"] is not None else False

        model = RandomForestClassifier
        reference_metric = "F1-Score"

    elif task_type == 'clustering':
        model = KMeans
        reference_metric = "Silhouette Score"

    return model, reference_metric, n_obsv, n_predictors, n_classes, is_multioutput

# 3 Get evaluation metrics
def evaluation_metrics(task_type):
    if task_type == 'regression':
        """ Regression """

        regression_metrics_names = ["Mean Squared Error (MSE)",
                                    "Root Mean Squared Error (RMSE)",
                                    "Mean Absolute Error (MAE)",
                                    "Mean Absolute Percentage Error (MAPE)",
                                    "R-Squared",
                                    "Adjusted R-Squared",
                                    "Explained Variance Score",
                                    ]

        def regression_evaluation_metrics(y_test, y_pred, n, p):
            # Calculating metrics
            MSE = mean_squared_error(y_test, y_pred)
            RMSE = np.sqrt(MSE)
            MAE = mean_absolute_error(y_test, y_pred)
            MAPE = mean_absolute_percentage_error(y_test, y_pred)
            R2 = r2_score(y_test, y_pred)
            
            # Adjusted R-Squared
            adj_r2 = 1 - (1 - R2) * ((n - 1) / (n - p - 1))
            
            # Explained Variance Score
            expl_var_score = explained_variance_score(y_test, y_pred)

            # Create a list of metric values in the same order as the dictionary keys
            metrics_values = [MSE, RMSE, MAE, MAPE, R2, adj_r2, expl_var_score]

            # Return all metrics as a tuple
            return metrics_values

        return [regression_metrics_names, regression_evaluation_metrics]

    elif task_type == 'classification':
        """ Classification """

        classification_metrics_names = ["Accuracy",
                            "Precision",
                            "Recall",
                            "F1-Score",
                            ]

        def classification_evaluation_metrics(y_test, y_pred, n_classes):
            # Average method for certain metrics
            if n_classes > 2:
                average = 'macro'
                
                precision = precision_score(y_test, y_pred, average=average, zero_division=np.nan)
                recall = recall_score(y_test, y_pred, average=average)
                f1_sc = f1_score(y_test, y_pred, average=average)

            else: # if n_classes == 2:
                average = 'binary'
                
                precision = precision_score(y_test, y_pred, average=average, zero_division=np.nan)
                recall = recall_score(y_test, y_pred, average=average)
                f1_sc = f1_score(y_test, y_pred, average=average)

            # accuracy
            accuracy = accuracy_score(y_test, y_pred)

            # Create a list of metric values in the same order as the dictionary keys
            metrics_values = [accuracy, precision, recall, f1_sc]

            return metrics_values

        return [classification_metrics_names, classification_evaluation_metrics]

    elif task_type == 'clustering':
        """ Clustering """

        clustering_metrics_names = ["Silhouette Score",
                                    "Davies-Bouldin Index",
                                    "Calinski-Harabasz Index",
                                    ]

        def clustering_evaluation_metrics(df, labels):
            # Silhouette score
            silhouette = silhouette_score(df, labels)       # Closer to 1 values suggest better-defined clusters.
            db_index = davies_bouldin_score(df, labels)     # A lower score is preferable
            ch_index = calinski_harabasz_score(df, labels)  # Higher is better

            # Create a list of metric values in the same order as the dictionary keys
            metrics_values = [silhouette, db_index, ch_index]

            return metrics_values

        return [clustering_metrics_names, clustering_evaluation_metrics]
    
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

# 4 Set hyperparameter bounds
def hyperparameters_bounds(model, random_state=42):
    # Model Name
    model_name = model.__name__

    if model_name == 'RandomForestRegressor':
        paras_bounds = [
            IntegerVar(lb=1, ub=100, name="n_estimators_paras"),
            StringVar(valid_sets=('squared_error', 'absolute_error', 'friedman_mse', 'poisson'), name="criterion_paras"),
            MixedSetVar(valid_sets=('none', 10, 50, 100), name="max_depth_paras"),
            IntegerVar(lb=2, ub=100, name="min_samples_split_paras"),                     # int in the range [2, inf) or a float in the range (0.0, 1.0]
            IntegerVar(lb=2, ub=100, name="min_samples_leaf_paras"),                      # int in the range [1, inf) or a float in the range (0.0, 1.0)
            FloatVar(lb=0., ub=0.5, name="min_weight_fraction_leaf_paras"),             # float in the range [0.0, 0.5]
            MixedSetVar(valid_sets=('none', 'sqrt', 'log2', 1, 5, 10, 50, 100), name="max_features_paras"),
            IntegerVar(lb=2, ub=100, name="max_leaf_nodes_paras"),                      # int in the range [2, inf)
            FloatVar(lb=1., ub=100., name="min_impurity_decrease_paras"),
            BoolVar(n_vars=1, name="bootstrap_paras"),                                  # `max_sample` cannot be set if `bootstrap=False`. Either switch to `bootstrap=True` or set `max_sample=None`
            BoolVar(n_vars=1, name="oob_score_paras"),                                  # Only available if bootstrap=True
            MixedSetVar(valid_sets=('none', 10, 50, 100), name="n_jobs_paras"),
            MixedSetVar(valid_sets=('none', random_state), name="random_state_paras"),  # Dependant towards bootstrap=True
            BoolVar(n_vars=1, name="warm_start_paras"),
            FloatVar(lb=0., ub=100., name="ccp_alpha_paras"),
            MixedSetVar(valid_sets=('none', 5, 10, 15), name="max_samples_paras"),      # `max_sample` cannot be set if `bootstrap=False`. Either switch to `bootstrap=True` or set `max_sample=None`

            # MixedSetVar(valid_sets=('none', -1, 0, -1), name="monotonic_cst_paras"),    # unspported when n_outputs_ > 1 (multioutput regression) or data has missing (NA) values
            # IntegerVar(lb=0, ub=3, name="verbose_paras"),                             # Irrelevant
        ]

    elif model_name == 'RandomForestClassifier':
        paras_bounds = [
            IntegerVar(lb=1, ub=100, name="n_estimators_paras"),
            StringVar(valid_sets=('gini', 'entropy', 'log_loss'), name="criterion_paras"),
            MixedSetVar(valid_sets=('none', 10, 50, 100), name="max_depth_paras"),
            IntegerVar(lb=2, ub=100, name="min_samples_split_paras"),                     # int in the range [2, inf) or a float in the range (0.0, 1.0]
            IntegerVar(lb=2, ub=100, name="min_samples_leaf_paras"),                      # int in the range [1, inf) or a float in the range (0.0, 1.0)
            FloatVar(lb=0., ub=0.5, name="min_weight_fraction_leaf_paras"),             # float in the range [0.0, 0.5]
            MixedSetVar(valid_sets=('none', 'sqrt', 'log2', 1, 5, 10, 50, 100), name="max_features_paras"),
            IntegerVar(lb=2, ub=100, name="max_leaf_nodes_paras"),                      # int in the range [2, inf)
            FloatVar(lb=1., ub=100., name="min_impurity_decrease_paras"),
            BoolVar(n_vars=1, name="bootstrap_paras"),                                  # `max_sample` cannot be set if `bootstrap=False`. Either switch to `bootstrap=True` or set `max_sample=None`
            BoolVar(n_vars=1, name="oob_score_paras"),                                  # Only available if bootstrap=True
            MixedSetVar(valid_sets=('none', 10, 50, 100), name="n_jobs_paras"),
            MixedSetVar(valid_sets=('none', random_state), name="random_state_paras"),  # Dependant towards bootstrap=True
            BoolVar(n_vars=1, name="warm_start_paras"),
            MixedSetVar(valid_sets=('none', 'balanced', 'balanced_subsample'), name="class_weight_paras"),
            FloatVar(lb=0., ub=100., name="ccp_alpha_paras"),
            MixedSetVar(valid_sets=('none', 5, 10, 15), name="max_samples_paras"),      # `max_sample` cannot be set if `bootstrap=False`. Either switch to `bootstrap=True` or set `max_sample=None`
            MixedSetVar(valid_sets=('none', -1, 0, 1), name="monotonic_cst_paras")      # not supported when n_classes > 2 (multiclass clf), n_outputs_ > 1 (multi-output), or data has missing values

            # IntegerVar(lb=0, ub=3, name="verbose_paras"),                             # Irrelevant
        ]

    elif model_name == 'KMeans':
        paras_bounds = [
            # FloatVar(lb=1e-5, ub=1e3, name="tol_paras"),
            # StringVar(valid_sets=('linear', 'poly', 'rbf', 'sigmoid'), name="kernel_paras"),
            StringVar(valid_sets=('lloyd', 'elkan'), name="algorithm_paras"),
            IntegerVar(lb=2, ub=20, name="n_clusters_paras"),
            IntegerVar(lb=100, ub=500, name="max_iter_paras"),
            MixedSetVar(valid_sets=('auto', 1, 5, 10, 15, 20), name="n_init_paras"),
            # BoolVar(n_vars=1, name="probability_paras"),
        ]

    paras_bounds_names = [] # List containing names of the hyperparameters
    for i, _ in enumerate(paras_bounds):
        paras_bounds_names.append(paras_bounds[i].name)  # Store each of parameter name (w/ "")

    return paras_bounds, paras_bounds_names

# n_classes=5
# 5 Dependencies handling
def dependencies_handling(all_decoded_paras,
                            self=None,
                            ml_model=None,
                            n_classes=None,
                            is_multioutput=None
                            ):
    
    # Model Name
    if self is None:
        ml_model_name = ml_model.__name__
        default_params_values = ml_model().get_params()
        # n_classes = 3
        # is_multioutput = is_multioutput
    else:
        ml_model_name = self.model.__name__
        default_params_values = self.model().get_params()
        n_classes = self.n_classes
        is_multioutput = self.is_multioutput
    
    # n_classes=10

    paras_names = list(all_decoded_paras.keys())

    if ml_model_name == 'RandomForestRegressor':

        required_keys = {"bootstrap", "max_samples", "oob_score"}

        if all(key in paras_names for key in required_keys):

            # Dep2: Handle the interdependency between bootstrap and max_samples
            if not all_decoded_paras["bootstrap"]:
                all_decoded_paras["max_samples"] = None  # Ensure max_samples is None if bootstrap=False
                all_decoded_paras["oob_score"] = False
        
        else:
            for required_key in required_keys:
                all_decoded_paras[required_key] = default_params_values[required_key]

    elif ml_model_name == 'RandomForestClassifier':

        required_keys = {"bootstrap", "max_samples", "oob_score", "class_weight", "warm_start", "monotonic_cst"}
        
        if all(key in paras_names for key in required_keys):
                
            # Dep2: Handle the interdependency between bootstrap and max_samples
            if not all_decoded_paras["bootstrap"]:
                all_decoded_paras["max_samples"] = None     # Ensure max_samples is None if bootstrap=False
                all_decoded_paras["oob_score"] = False

            # Dep3: Handle monotonic constraint
            if n_classes > 2 or is_multioutput:
                all_decoded_paras["monotonic_cst"] = None   # set monotonic_cst to None for multiclass classification or multi-output
            
            # Dep4: class_weight & warm_start
            if all_decoded_paras["class_weight"] in ('balanced', 'balanced_subsample'):
                all_decoded_paras["warm_start"] = False
        
        else:
            for required_key in required_keys:
                all_decoded_paras[required_key] = default_params_values[required_key]

    return all_decoded_paras

# 6 Define problem class
class OptimizedProblem(Problem):
    def __init__(
                    self,
                    bounds=None,
                    minmax="max",
                    data=None,
                    model=None,
                    task_type=None,
                    paras_bounds_names=None,
                    # n_classes=None,
                    # is_multioutput=None,
                    **kwargs
                ):
        self.data = data       
        self.model = model
        self.task_type = task_type
        self.paras_bounds_names = paras_bounds_names
        # self.n_classes = n_classes
        # self.is_multioutput = is_multioutput

        self.all_decoded_paras = {}
        self.encoders = {}

        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, x):
        task_type = self.task_type
        all_decoded_paras = self.all_decoded_paras
        original_paras = {}

        x_decoded = self.decode_solution(x)

        # print(self.paras_bounds_names)
        for paras_name in self.paras_bounds_names:

            original_paras[paras_name] = x_decoded[paras_name]

            all_decoded_paras[paras_name[:-6]] = None if original_paras[paras_name] == 'none' else original_paras[paras_name]

        # Decoded paras (dict) after handling dependecies
        all_decoded_paras = dependencies_handling(all_decoded_paras, self=self)

        # Defining the model and assigning hyperparameters
        ml_model = self.model(**all_decoded_paras)  

        # Supervised tasks
        if task_type in ('regression', 'classification'):

            # Fit the model
            ml_model.fit(self.data["X_train"], self.data["y_train"])

            # Make the predictions
            y_predict = ml_model.predict(self.data["X_test"])

            # MSE for Regression
            if task_type == 'regression':
                return mean_squared_error(self.data["y_test"], y_predict)

            # F1-Score for Classification
            elif task_type == 'classification':
                return f1_score(self.data["y_test"], y_predict, average='macro')

        # Unsupervised tasks (Clustering)
        elif task_type == 'clustering':
            
            # Fit the model
            ml_model.fit_predict(self.data["X"])
            
            # Make the predictions
            labels = ml_model.fit_predict(self.data["X"])
            
            # Silhouette Score for Clustering
            return silhouette_score(self.data["X"], labels)

# 7 Re-fit and re-predict using best parameters
# 7.1 Decode best paras
def decode_best_paras(ml_model, best_paras_opt, n_classes, is_multioutput):

    best_paras_decoded = {}

    # Iterate over all items in the dictionary
    for key, value in best_paras_opt.items():

        # Remove the '_paras' suffix from the key
        # and check if the value is 'none', and set to None if so
        best_paras_decoded[key[:-6]] = None if value == 'none' else value

    # # Debug: Check the dictionary after modification
    # print(f"Decoded best parameters before handling dependencies: {best_paras_decoded}")

    # Apply dependency handling (ensure this doesn't overwrite the decoded values)
    best_paras_decoded = dependencies_handling(best_paras_decoded,
                                                self=None,
                                                ml_model=ml_model,
                                                n_classes=n_classes,
                                                is_multioutput=is_multioutput
                                                )

    # # Debug: Check the dictionary after dependency handling
    # print(f"Decoded best parameters after handling dependencies: {best_paras_decoded}")

    return best_paras_decoded

# 7.2 Re-fit and re-predict
def optimized_fit_predict(model,
                            paras,
                            data,
                            task_type,
                            eval_metrics,
                            label = None,
                            n_obsv = None,
                            n_predictors = None,
                            n_classes = None,
                            is_multioutput = None,
                            ):
    
    ml_model = model(**paras)

    if task_type in ('regression', 'classification'):

        n_obsv = len(data["y_test"]) if n_obsv is None else n_obsv                  # Number of observations
        n_predictors = data["X_test"].shape[1] if n_predictors is None else n_predictors  # Number of predictors (features)
        n_classes = len(np.unique(data["y_train"])) if n_classes is None else n_classes
        is_multioutput = len(data["y_train"].shape) > 1 and data["y_train"].shape[1] > 1 if data["y_train"] is not None and is_multioutput is None else False        

        # Fit the model
        ml_model.fit(data["X_train"], data["y_train"])

        # Make the predictions
        y_predict = ml_model.predict(data["X_test"])

        if task_type == 'regression':
            metrics = eval_metrics(data["y_test"], y_predict, n_obsv, n_predictors)

        elif task_type == 'classification':
            metrics = eval_metrics(data["y_test"], y_predict, n_classes)

    elif task_type == 'clustering':
        
        # Fit the model
        ml_model.fit_predict(data["X"])
        
        # Make the predictions
        labels = ml_model.fit_predict(data["X"])
        
        metrics = eval_metrics(data["X"], labels)
    
    return [ml_model, metrics]

# Evaluation
def evaluate(data, task_type):

    # Assign data into specified cases
    data_dict = create_data_dict(data, task_type)

    # Model and necessary variable(s)
    model, reference_metric, n_obsv, n_predictors, n_classes, is_multioutput = model_and_variables(data_dict, task_type)

    # For evaluation
    metrics_names, eval_metrics = evaluation_metrics(task_type)

    # Setting the min-max value
    minmax_val = "min" if reference_metric in ["Mean Squared Error (MSE)",
                                                "Root Mean Squared Error (RMSE)",
                                                "Mean Absolute Error (MAE)",
                                                "Mean Absolute Percentage Error (MAPE)",
                                                "Davies-Bouldin Index",
                                                ] else "max"

    # Getting hyperparameter bounds and names
    paras_bounds, paras_bounds_names = hyperparameters_bounds(model, random_state=42)

    epoch = 10
    pop_size = 10

    # Assigning Metaheursitic Optimizer
    optimizers = [
        # ACOR.OriginalACOR(epoch=epoch, pop_size=pop_size, sample_count = 25, intent_factor = 0.5, zeta = 1.0),
        GA.BaseGA(epoch=epoch, pop_size=pop_size, pc=0.9, pm=0.05, selection="tournament", k_way=0.4, crossover="multi_points", mutation="swap"), # Epoch & pop_size minimal 10
        # PSO.OriginalPSO(epoch=epoch, pop_size=pop_size, c1=2.05, c2=2.05, w=0.4),
        SA.OriginalSA(epoch=epoch, pop_size=pop_size, temp_init=100, step_size=0.1),
    ]

    # List for containing evaluation values
    metaopt_name = []
    metaopt_object = []
    ml_models = []
    best_metrics = []
    time_taken = []

    # Evaluation through iteration
    for optimizer in optimizers:

        #  Defining the problem class
        problem = OptimizedProblem(bounds=paras_bounds,
                                    minmax=minmax_val,
                                    data=data_dict,
                                    model=model,
                                    task_type=task_type,
                                    paras_bounds_names=paras_bounds_names,
                                    n_classes = n_classes,
                                    is_multioutput = is_multioutput,
                                    )

        # Time monitoring and optimization process
        start = time.perf_counter()
        optimizer.solve(problem)
        end = time.perf_counter() - start

        best_paras = optimizer.problem.decode_solution(optimizer.g_best.solution)

        best_paras_decoded = decode_best_paras(model, best_paras, n_classes, is_multioutput)
        
        best_ml_model, best_metrics_opt = optimized_fit_predict(model = model,
                                                                paras = best_paras_decoded,
                                                                data = data_dict,
                                                                task_type = task_type,
                                                                eval_metrics = eval_metrics,
                                                                n_classes = n_classes,
                                                                is_multioutput = is_multioutput,
                                                                n_obsv = n_obsv,
                                                                n_predictors = n_predictors,
                                                                )

        metaopt_name.append(optimizer.__class__.__name__)
        metaopt_object.append(optimizer)
        ml_models.append(best_ml_model)
        best_metrics.append(best_metrics_opt)
        time_taken.append(end)

        print(f"Best agent: {optimizer.g_best}")
        print(f"Best solution: {optimizer.g_best.solution}")
        print(f"Best {reference_metric}: {optimizer.g_best.target.fitness}")
        print(f"Best parameters: {best_paras}\n")        

    # Final result
    result_df = pd.DataFrame ({
        "Metaheuristic Optimizer (Name)" : metaopt_name,
        "Metaheuristic Optimizer (Object)" : metaopt_object,
        "Machine Learning Model (object)" : ml_models,
        **{metric: values for metric, values in zip(metrics_names, zip(*best_metrics))},
        "Time taken (s)" : time_taken,
    })

    # Save the trained model
    ascending = None
    if minmax_val == "max":
        ascending = False
    else:
        ascending = True

    best_ml_model = result_df.sort_values(by=reference_metric, ascending=ascending).iloc[0,2]
    joblib.dump(best_ml_model, f'Best_{best_ml_model.__class__.__name__}.pkl')

    return result_df, data_dict, reference_metric