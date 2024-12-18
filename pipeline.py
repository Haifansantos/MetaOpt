import streamlit as st

import os
import time
import datetime
import joblib
import mealpy

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

### Import modules
# import load_data
# import preprocessing
# import eda
# import evaluation
# import visualization

# from evaluation import create_data_dict
# from visualization import feature_importance_visualization

#######################################  PREPROCESSING  ############################################

# 1 Data Cleaning
def data_cleaning(data, task_type, label=None): #, alpha, threshold):
    # If there's no label input
    if label is None:
        label = data.columns[-1]
    
    # NA
    data.dropna()

    # Remove duplicates
    data.drop_duplicates()

    # Handle outliers (IQR)
    data = handling_outliers(data) #, alpha) # requires alpha value (default: 0.05)

    # Handle ID-like column
    data = handling_id_cols(data, label) #, threshold) # requires threshold (default: 0.99)
    
    return data

# 1.1 def handling_outliers(data, alpha=0.05):
def handling_outliers(data, alpha=0.05):
    """
    Handles outliers by dropping the records containing outlier(s).

    Parameters:
        data (pd.DataFrame): The input dataframe.
        alpha (float): The alpha value as the threshold to determine the normality of a data (default: 0.05).
    
    Returns:
        pd.DataFrame: The dataframe with records containing outlier(s) removed.
    """
    numeric_features = data.select_dtypes(include=[np.number]).columns # Retrieving numeric features from the dataset
    outlier_indices = set()  # Use a set to store unique indices of outlier rows

    # Iterating through each numeric features
    for numeric_feature in numeric_features:
        numeric_feature_data = data[numeric_feature]          # Store the selected column into an object called "numeric_feature_data"

        _, p = shapiro(numeric_feature_data)                # Retrieving p velue evaluated from the Shapiro-Wilk Statistical test

        if p > alpha:
            pass                            # Skipping normally distributed numeric_feature_data
        else:
            q1 = numeric_feature_data.quantile(0.25)        # Retrieving the value of the 1st quantile (25%)
            q3 = numeric_feature_data.quantile(0.75)        # Retrieving the value of the 3rd quantile (75%)
            iqr = q3 - q1                   # Interquartile range

            # Define bounds for outliers
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Identify outlier indices
            outliers = data[(numeric_feature_data < lower_bound) | (numeric_feature_data > upper_bound)].index
            outlier_indices.update(outliers)  # Add these indices to the set
    
    # # Testing
    # print("Index(es) of outlier-contained records:", outlier_indices, '\n')

    # Drop rows containing outliers
    return data.drop(index=outlier_indices)

# 1.2 Handling ID-like columns
def handling_id_cols(data, label=None, threshold=0.99999):
    """
    Handles id-like columns by dropping those with high cardinality.

    Parameters:
        data (pd.DataFrame): The input dataframe.
        label (str): The label column to exclude from removal (default: None).
        threshold (float): The cardinality threshold to identify id-like columns (default: 0.99).
    
    Returns:
        pd.DataFrame: The dataframe with id-like columns removed.
    """
    # If there's no label input
    if label is None:
        label = data.columns[-1]

    # Identify id-like columns
    id_like_cols = [
        col for col in data.columns
        if data[col].nunique() / len(data) > threshold and col != label
    ]

    # # Testing
    # print("ID-like column:", id_like_cols, '\n')

    # Drop id-like columns
    return data.drop(columns=id_like_cols)

# 2 Data transformation
# Dev purps
scaler = StandardScaler()

def data_transformation(data, task_type, label=None):

    # For supervised task
    if task_type in ('regression', 'classification'):

        # If there's no label input
        if label is None:
            label = data.columns[-1]

        # All column name
        colnames = data.drop(columns=label).columns            

        # Retirieving categorical feature names
        feature_names = data.drop(columns=label).select_dtypes(include=["object"]).columns

        # Feature-target split
        X, y = feature_target_split(data, label)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)#, stratify=y if task_type == 'classification' else None)

        # Encoding (X_train, X_test, y_train, y_test)
        data = feature_target_encoding(X_train, X_test, y_train, y_test, task_type, feature_names)

        # Feature Scaling
        data[0] = pd.DataFrame(scaler.fit_transform(data[0]), columns=colnames)    # Scaling X_train
        data[1] = pd.DataFrame(scaler.transform(data[1]), columns=colnames)        # Scaling X_test

    # For unsupervised task
    elif task_type == 'clustering':

        # All column name
        colnames = data.columns
        
        # Encoding
        data = feature_encoding(data)

        # Scaling
        data = pd.DataFrame(scaler.fit_transform(data), columns=colnames)

    return data

# 2.1 Feature-target split
def feature_target_split(data, label=None):
    # If there's no label input
    if label is None:
        label = data.columns[-1]
    
    if label:
        X = data.drop(columns=label)
        y = data[label]
    else:
        X = data.iloc[:,:-1]
        y = data.iloc[:,-1]

    return X, y

# 2.2 Feature-target encoding
# Dev purps
all_encoder = {}

def feature_target_encoding(X_train, X_test, y_train, y_test, task_type, feature_names=None):
    
    if feature_names is None:
        feature_names = X_train.select_dtypes(exclude=[np.number]).columns
    
    # Instantiating encoders dicationary
    feature_encoders = {}
    target_encoder = None
    
    # Encoding each column through iteration
    for feature in feature_names:

        # Instantiate LabelEncoder object
        fe = LabelEncoder()

        # Fit and transform the features of the train set
        X_train[feature] = fe.fit_transform(X_train[feature])

        # Fit and transform the features of the test set
        X_test[feature] = fe.transform(X_test[feature])

        # Store the fitted feature encoders
        feature_encoders[feature] = fe
    
    if task_type == 'classification':
        # Instantiate the encoder object for target
        te = LabelEncoder()

        # Encoding the target of the train set
        y_train = te.fit_transform(y_train)

        # Encoding the target of the test set
        y_test = te.transform(y_test)

        # Store the fitted target encoder
        target_encoder = te

    # print(target_encoder)

    # Store all the fitted encoders
    all_encoder['feature_encoders'] = feature_encoders
    all_encoder['target_encoder'] = target_encoder

    return [X_train, X_test, y_train, y_test]

# 2.3 Feature encoding (unsupervised)
# Dev
all_encoder = {}

def feature_encoding(data, feature_names=None):
    if not feature_names:
        feature_names = data.select_dtypes(exclude=[np.number]).columns
    
    # Instantiating encoders dicationary
    feature_encoders = {}

    # Encoding each column through iteration
    for feature in feature_names:

        # Instantiate LabelEncoder object
        fe = LabelEncoder()

        # Fit and transform the features of the train set
        data[feature] = fe.fit_transform(data[feature])

        # Store the fitted feature encoders
        feature_encoders[feature] = fe
    
    # Store all the fitted encoders
    all_encoder['feature_encoders'] = feature_encoders

    return data

#######################################       EDA       ############################################

# Save plot
def save_plot(folder_path, plot_title):
    file_path = os.path.join(folder_path, plot_title)
    
    if os.path.exists(file_path):
        pass
    else:
        plt.savefig(file_path, dpi=300, bbox_inches='tight')

# # EDA for dashboard
# def eda_visualization(data, task_type, label=None):
#     """
#     Displaying the Dataset Information, Dataset description, and
#     Plotting a pairplot, Boxplot, and Heatmap of the correlation matrix of the features

#     Parameters
#     ------
#     data  : Pandas DataFrame
#         DataFrame from which the Info, Desc, and Pairplot is retrieved
#     """

#     # Define the folder path
#     folder_path = './eda_plots'
#     os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist

#     if isinstance(data, list) and len(data) == 4:
#         # If there's no input label
#         if label is None:
#             if task_type == 'regression':
#                 label = df_reg.columns[-1]
#             if task_type == 'classification':
#                 label = df_clf.columns[-1]

#         # Combine train and test features
#         X_combined = pd.concat([data[0], data[1]], axis=0).reset_index(drop=True)

#         # Combine train and test labels
#         y_train_series = pd.Series(data[2], name=label).reset_index(drop=True)
#         y_test_series = pd.Series(data[3], name=label).reset_index(drop=True)
#         y_combined = pd.concat([y_train_series, y_test_series], axis=0).reset_index(drop=True)

#         # Combine features and labels into a single DataFrame
#         data = pd.concat([X_combined, y_combined], axis=1)

#     # Plotting the Pairwise relationship in the dataset
#     pairplot_title = "Pairwise relationship plot"
#     sns.pairplot(data)
#     plt.gcf().suptitle(pairplot_title, y=1.02)
#     st.pyplot(plt)
#     plt.figure()

#     cols = st.columns(data.columns.nunique())
#     # Plotting the Boxplot for all the columns in the dataset
#     for column_name in data.columns:
#         boxplot_title = f"Boxplot for the {column_name} column"
#         sns.boxplot(data[column_name])
#         plt.title(boxplot_title)
#         st.pyplot(plt)
#         plt.figure()

#     # Displaying correlation matrix of the features in the dataset
#     corr_mtx_title = "Correlation Matrix"
#     matrix = data.corr()
#     sns.heatmap(matrix, cmap="Blues", annot=True)
#     plt.title(corr_mtx_title)
#     st.pyplot(plt)

# eda visualization
def eda_visualization(data, task_type, label=None):
    """
    Displaying the Dataset Information, Dataset description, and
    Plotting a pairplot, Boxplot, and Heatmap of the correlation matrix of the features

    Parameters
    ------
    data  : Pandas DataFrame
        DataFrame from which the Info, Desc, and Pairplot is retrieved 
    """

    if isinstance(data, list) and len(data) == 4:
        # If there's no input label
        if label is None:
            if task_type == 'regression':
                label = df_reg.columns[-1]
            if task_type == 'classification':
                label = df_clf.columns[-1]

        # Combine train and test features
        X_combined = pd.concat([data[0], data[1]], axis=0).reset_index(drop=True)

        # Combine train and test labels
        y_train_series = pd.Series(data[2], name=label).reset_index(drop=True)
        y_test_series = pd.Series(data[3], name=label).reset_index(drop=True)
        y_combined = pd.concat([y_train_series, y_test_series], axis=0).reset_index(drop=True)

        # Combine features and labels into a single DataFrame
        data = pd.concat([X_combined, y_combined], axis=1)

    # Define the folder path
    folder_path = './eda_plots'
    os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist
    
    # Plotting the Pairwise relationship in the dataset
    pairplot_title = f"Pariwise relationship plot ({task_type})"
    
    sns.pairplot(data)
    plt.gcf().suptitle(pairplot_title, y=1.02)
    
    save_plot(folder_path, pairplot_title)
    plt.figure()

    # Plotting the Boxplot for all the columns in the dataset
    for column_name in data.columns:
        boxplot_title = f"Boxplot for the {column_name} column ({task_type})"

        sns.boxplot(data[column_name])
        plt.title(boxplot_title)
        
        save_plot(folder_path, boxplot_title)
        plt.figure()

    # Displaying correlation matrix of the features in the dataset
    corr_mtx_title = f"Correlation Matrix ({task_type})"
    matrix = data.corr()

    sns.heatmap(matrix, cmap="Blues", annot=True)
    plt.title(corr_mtx_title)
    
    save_plot(folder_path, corr_mtx_title)
    plt.show()

#######################################   EVALUATION    ############################################

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
    if hasattr(self, 'encoders') and isinstance(self.encoders, dict):
    self.encoders['feature_encoders'] = feature_encoders
    self.encoders['target_encoder'] = target_encoder
else:
    all_encoder = {}
    all_encoder['feature_encoders'] = feature_encoders
    all_encoder['target_encoder'] = target_encoder

    
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

#######################################    FINAL VIZ    ############################################

# 1 Compute feature importance
def compute_feature_importance(best_ml_model, data_dict, label=None):

    if hasattr(best_ml_model, "coef_"):  # Linear models
        feature_importance = np.abs(best_ml_model.coef_[0])

    elif hasattr(best_ml_model, "feature_importances_"):  # Tree-based models
        feature_importance = best_ml_model.feature_importances_

    else:  # Model-agnostic
        # data = None
        # label = None
        # X = None
        # y = None

        data = data_dict["X"]

        label = "Cluster"

        data[label] = best_ml_model.labels_
        
        # Feature-target split
        X = data.drop(columns=label)
        y = data[label]

        # Train-test split
        # X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=label), data[label], test_size=0.2, random_state=42)#, stratify=y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)#, stratify=y)

        # Train the model (Random Forest)
        fi_model = RandomForestClassifier(random_state=42)
        fi_model.fit(X_train, y_train)

        # Compute permutation importance
        perm_importance = permutation_importance(
            fi_model, X_test, y_test, scoring='accuracy', random_state=42
        )

        # Extract importance scores
        importance_df = pd.DataFrame(
            {
                "Feature": X.columns,
                "Importance Mean": perm_importance.importances_mean,
                "Importance Std": perm_importance.importances_std,
            }
        ).sort_values(by="Importance Mean", ascending=False)
        
        return importance_df

    # Linear or Tree-based models
    importance_df = pd.DataFrame({
        'Feature': data_dict["X_train"].columns,
        'Importance': feature_importance
    })

    return importance_df

# # 2 Visualze feature importance for Dashboard
# def feature_importance_visualization(data, task_type, optimizer_name=None):
    
#     # Define the folder path
#     folder_path = 'feature_importance_plots'
#     os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist

#     if task_type == 'clustering':
#         plot_title = f"Feature Importance (Permutation) - {optimizer_name}"

#         # Visualization of Permutation Importance
#         plt.figure(figsize=(10, 6))
#         plt.barh(
#             data["Feature"], 
#             data["Importance Mean"], 
#             xerr=data["Importance Std"]
#         )
#         plt.gca().invert_yaxis()  # Flip the order for better readability
#         plt.xlabel("Permutation Importance")
#         plt.title(plot_title)
#         plt.tight_layout()
#         st.pyplot(plt)
#         # plt.show()
#     else:
#         plot_title = f"Feature Importance ({optimizer_name})"
#         # Sort features by importance
#         data = data.sort_values(by='Importance', ascending=False)

#         # Visualize the feature importance
#         plt.figure(figsize=(10, 6))
#         sns.barplot(x='Importance', y='Feature', data=data, palette='viridis')
#         plt.title(plot_title, fontsize=16)
#         plt.xlabel("Importance", fontsize=12)
#         plt.ylabel("Features", fontsize=12)
#         plt.tight_layout()
#         st.pyplot(plt)
#         # plt.show()
    
#     # save_plot(folder_path, plot_title)

# 2 Visualze feature importance
def feature_importance_visualization(data, task_type, optimizer_name=None):
    
    # Define the folder path
    folder_path = './feature_importance_plots'
    os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist

    if task_type == 'clustering':
        plot_title = f"Feature Importance (Permutation) - {optimizer_name}"

        # Visualization of Permutation Importance
        plt.figure(figsize=(10, 6))
        plt.barh(
            data["Feature"], 
            data["Importance Mean"], 
            xerr=data["Importance Std"]
        )
        plt.gca().invert_yaxis()  # Flip the order for better readability
        plt.xlabel("Permutation Importance")
        plt.title(plot_title)
        plt.tight_layout()
        plt.show()
    else:
        plot_title = f"Feature Importance ({optimizer_name})"
        # Sort features by importance
        data = data.sort_values(by='Importance', ascending=False)

        # Visualize the feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=data, hue=data.iloc[:,-1], legend=False)#, palette='viridis')
        plt.title(plot_title, fontsize=16)
        plt.xlabel("Importance", fontsize=12)
        plt.ylabel("Features", fontsize=12)
        plt.tight_layout()
        plt.show()
    
    save_plot(folder_path, plot_title)

####################################################################################################

class MetaheuristicPipeline:
    def __init__(self, filepath, task_type, label):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier()
    
    def load_data(self, filepath):
        # Load dataset
        return pd.read_csv(filepath)
    
    # def preprocess(self, data, label):
    #     # Split features and labels
    #     X = data.drop(columns=[label])
    #     y = data[label]
    #     # Scale features
    #     X_scaled = self.scaler.fit_transform(X)
    #     return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Preprocessing
    def preprocessing(data, task_type, label=None):

        # If there's no label input
        if label is None:
            label = data.columns[-1]

        data = data_cleaning(data, task_type, label)
        data = data_transformation(data, task_type, label)
        
        return data
    
    # def train(self, X_train, y_train):
    #     # Train model
    #     self.model.fit(X_train, y_train)

    # eda visualization
    def eda_visualization(data, task_type, label=None, self=None):
        """
        Displaying the Dataset Information, Dataset description, and
        Plotting a pairplot, Boxplot, and Heatmap of the correlation matrix of the features

        Parameters
        ------
        data  : Pandas DataFrame
            DataFrame from which the Info, Desc, and Pairplot is retrieved 
        """

        if isinstance(data, list) and len(data) == 4:
            # If there's no input label
            if label is None:
                if task_type == 'regression':
                    label = df_reg.columns[-1]
                if task_type == 'classification':
                    label = df_clf.columns[-1]

            # Combine train and test features
            X_combined = pd.concat([data[0], data[1]], axis=0).reset_index(drop=True)

            # Combine train and test labels
            y_train_series = pd.Series(data[2], name=label).reset_index(drop=True)
            y_test_series = pd.Series(data[3], name=label).reset_index(drop=True)
            y_combined = pd.concat([y_train_series, y_test_series], axis=0).reset_index(drop=True)

            # Combine features and labels into a single DataFrame
            data = pd.concat([X_combined, y_combined], axis=1)

        # Define the folder path
        folder_path = './eda_plots'
        os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist
        
        # Plotting the Pairwise relationship in the dataset
        pairplot_title = f"Pariwise relationship plot ({task_type})"
        
        sns.pairplot(data)
        plt.gcf().suptitle(pairplot_title, y=1.02)
        
        save_plot(folder_path, pairplot_title)
        plt.figure()

        # Plotting the Boxplot for all the columns in the dataset
        for column_name in data.columns:
            boxplot_title = f"Boxplot for the {column_name} column ({task_type})"

            sns.boxplot(data[column_name])
            plt.title(boxplot_title)
            
            save_plot(folder_path, boxplot_title)
            plt.figure()

        # Displaying correlation matrix of the features in the dataset
        corr_mtx_title = f"Correlation Matrix ({task_type})"
        matrix = data.corr()

        sns.heatmap(matrix, cmap="Blues", annot=True)
        plt.title(corr_mtx_title)
        
        save_plot(folder_path, corr_mtx_title)
        plt.show()
    
    # def evaluate(self, X_test, y_test):
    #     # Evaluate model
    #     predictions = self.model.predict(X_test)
    #     return accuracy_score(y_test, predictions)

    # Evaluation
    def evaluate(data, task_type, self=None):

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

    # Visualization
    # for result_df, data_dict, task_type in zip(result_df_list, data_dict_list, task_type_list):

    #     # Optimizers names
    #     optimizers_names = result_df[result_df.columns[0]]

    #     # Machine Learning Models
    #     ml_models = result_df.iloc[:,2]

    def final_visualization(result_df, task_type, data_dict, self=None):
        optimizers_names = result_df.iloc[:,0]
        ml_models = result_df.iloc[:,2]

        # Iterate through each model
        for optimizer_name, best_ml_model in zip(optimizers_names, ml_models):

            # Compute the feature importance
            feature_importance_data = compute_feature_importance(best_ml_model, data_dict)
            
            # Generate the visualization
            feature_importance_visualization(feature_importance_data, task_type, optimizer_name)
        
##############################################################################################################

# filepath = "C:/Users/user/MetaOpt/data/Iris.csv"
# task_type = "classification"
# label = None

# pipeline = MetaheuristicPipeline()

# data = pipeline.load_data(filepath)

# processed_data = pipeline.preprocessing(task_type, label)

# eda_visualization(processed_data, task_type, label)

# result_df, data_dict, reference_metric = pipeline.evaluation(data, task_type)

# pipeline.final_visualization(result_df, task_type, data_dict)

# import sys

# print(f"Python version: {sys.version}")

