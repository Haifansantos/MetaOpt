# --- Start of libraries.py ---
import os
import time
import datetime
import joblib
import mealpy

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from google.cloud import storage 
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
# --- End of libraries.py ---

# --- Start of preprocessing.py ---
# import os
# import time
# import datetime
# import joblib

# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# from scipy.stats import shapiro

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from sklearn.cluster import KMeans
# from sklearn.inspection import permutation_importance


# from sklearn.metrics import (
#     mean_squared_error,
#     mean_absolute_error,
#     mean_absolute_percentage_error,
#     r2_score,
#     explained_variance_score,

#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,

#     silhouette_score,
#     davies_bouldin_score,
#     calinski_harabasz_score,
# )

# from mealpy import FloatVar, StringVar, IntegerVar, BoolVar, MixedSetVar, Problem, ACOR, GA, PSO, SA

# Data Preprocessing

"""
Variables needed:
- scaler (StandarScaler)
- all_encoder (dict; to store all the encoder[s])

Returns:
- Preprocessed dataframes (list for supervised task and pd.DataFrame for unsupervised task)
"""

# 1 Data Cleaning
# - - dropna
# - - drop_duplicates
# 1.1 Handling outliers
# 1.2 Handling ID-like columns

# 2 Data Transformation
# - - check task_type
# - - supervised
# - - - check label
# - - - get feature names
# - 2.1 feature-target split
# - - - Train-test split
# - 2.2 feature-target encoding
# - - - feature scaling
# - - unsupervised
# - - - get all column names
# - 2.3 feature encoding
# - - - feature scaling

# 1 Data Cleaning
def data_cleaning(data, task_type, label=None):
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

def data_transformation(self, data, task_type, label=None):
    # For supervised task
    if task_type in ('regression', 'classification'):

        # If there's no label input
        if label is None:
            label = data.columns[-1]
        colnames = data.drop(columns=label).columns            

        # Retirieving categorical feature names
        feature_names = data.drop(columns=label).select_dtypes(include=["object"]).columns
        X, y = feature_target_split(data, label)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        data = feature_target_encoding(self, X_train, X_test, y_train, y_test, task_type, feature_names)

        # Feature Scaling
        data[0] = pd.DataFrame(scaler.fit_transform(data[0]), columns=colnames)    # Scaling X_train
        data[1] = pd.DataFrame(scaler.transform(data[1]), columns=colnames)        # Scaling X_test

    # For unsupervised task
    elif task_type == 'clustering':
        colnames = data.columns
        data = feature_encoding(self, data)
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
def feature_target_encoding(self, X_train, X_test, y_train, y_test, task_type, feature_names=None):
    
    if feature_names is None:
        feature_names = X_train.select_dtypes(exclude=[np.number]).columns
    
    # Instantiating encoders dicationary
    feature_encoders = {}
    target_encoder = None
    
    # Encoding each column through iteration
    for feature in feature_names:
        fe = LabelEncoder()
        X_train[feature] = fe.fit_transform(X_train[feature])
        X_test[feature] = fe.transform(X_test[feature])
        feature_encoders[feature] = fe
    
    if task_type == 'classification':
        te = LabelEncoder()
        y_train = te.fit_transform(y_train)
        y_test = te.transform(y_test)
        target_encoder = te

    # Store all the fitted encoders
    if self is None:
        all_encoder = {}
        all_encoder['feature_encoders'] = feature_encoders
        all_encoder['target_encoder'] = target_encoder
    else:
        self.encoders['feature_encoders'] = feature_encoders
        self.encoders['target_encoder'] = target_encoder

    return [X_train, X_test, y_train, y_test]

# 2.3 Feature encoding (unsupervised)
# Dev
def feature_encoding(self, data, feature_names=None):
    if not feature_names:
        feature_names = data.select_dtypes(exclude=[np.number]).columns

    feature_encoders = {}
    for feature in feature_names:
        fe = LabelEncoder()
        data[feature] = fe.fit_transform(data[feature])
        feature_encoders[feature] = fe
    
    # Store all the fitted encoders
    if self is None:
        all_encoder = {}
        all_encoder['feature_encoders'] = feature_encoders
    else:
        self.encoders['feature_encoders'] = feature_encoders

    return data

# # Preprocessing
# def preprocessing(data, task_type, label=None):

#     # If there's no label input
#     if label is None:
#         label = data.columns[-1]

#     data = data_cleaning(data, task_type, label)
#     data = data_transformation(data, task_type, label, self)
    
#     return data
# --- End of preprocessing.py ---

# --- Start of eda.py ---
# import streamlit as st

# import os
# import time
# import datetime
# import joblib

# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# from scipy.stats import shapiro

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from sklearn.cluster import KMeans
# from sklearn.inspection import permutation_importance


# from sklearn.metrics import (
#     mean_squared_error,
#     mean_absolute_error,
#     mean_absolute_percentage_error,
#     r2_score,
#     explained_variance_score,

#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,

#     silhouette_score,
#     davies_bouldin_score,
#     calinski_harabasz_score,
# )

# from mealpy import FloatVar, StringVar, IntegerVar, BoolVar, MixedSetVar, Problem, ACOR, GA, PSO, SA

# Exploratory Data Analysis (EDA)

"""
Variables needed:
- None

Returns:
- None (images are stored within './eda_plots/')
"""

# Save plot func
# eda_visualization

# Save plot
def save_plot(plot_title, folder_path, is_cloud_env, bucket_name=None,):
    """
    Save the plot directly to a cloud storage bucket or locally.

    Parameters
    ----------
    plot_title : str
        Title of the plot file.
    folder_path : str
        Logical folder path for saving plots.
    is_cloud_env : bool
        Indicates if the environment is cloud-based.
    bucket_name : str, optional
        Cloud storage bucket name.
    folder_prefix : str, optional
        Folder path or prefix inside the bucket to save files.
    """
    # Ensure folder_path is clean
    folder_path = folder_path.strip("/")  # Pastikan tidak ada slash di awal
    plot_title = plot_title.strip("/")   # Sama untuk nama file

    # Construct paths
    full_file_path = f"{folder_path}/{plot_title}"

    if is_cloud_env:
        # Save plot to a temporary file
        temp_file_path = f"{plot_title}.png"
        plt.savefig(temp_file_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Upload to the cloud bucket
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        folder_prefix = "eda_plots"
        blob = bucket.blob(f"{folder_prefix}/{full_file_path}")
        blob.upload_from_filename(temp_file_path)

        # Delete the temporary file after upload
        os.remove(temp_file_path)
        print(f"File uploaded directly to {bucket_name}/{folder_prefix}/{full_file_path}")
    else:
        # Save locally
        os.makedirs(folder_path, exist_ok=True)
        local_file_path = os.path.join(folder_path, plot_title)
        plt.savefig(local_file_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Plot saved locally: {local_file_path}")


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
# --- End of eda.py ---

# --- Start of evaluation.py ---
# import os
# import time
# import datetime
# import joblib

# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# from scipy.stats import shapiro

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from sklearn.cluster import KMeans
# from sklearn.inspection import permutation_importance


# from sklearn.metrics import (
#     mean_squared_error,
#     mean_absolute_error,
#     mean_absolute_percentage_error,
#     r2_score,
#     explained_variance_score,

#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,

#     silhouette_score,
#     davies_bouldin_score,
#     calinski_harabasz_score,
# )

# from mealpy import FloatVar, StringVar, IntegerVar, BoolVar, MixedSetVar, Problem, ACOR, GA, PSO, SA

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
class TaskInfo:
    def __init__(self, data, task_type):
        self.task_type = task_type
        self.data = data

        self.model = self._get_model_and_variables()[0]
        self.reference_metric = self._get_model_and_variables()[1]
        self.n_obsv = self._get_model_and_variables()[2] if task_type == "regression" else None
        self.n_predictors = self._get_model_and_variables()[3] if task_type == "regression" else None
        self.n_classes = self._get_model_and_variables()[4] if task_type == "classification" else None
        self.is_multioutput = self._get_model_and_variables()[5] if task_type == "classification" else None

        self.metrics_names = self._get_evaluation_metrics()[0]
        self.eval_metrics = self._get_evaluation_metrics()[1]

    def _get_model_and_variables(self):
        """
        To get model, reference metric
        - Regression: n_obsv, n_predictors
        - Classification: n_classes, is_multioutput
        """
        data = self.data
        task_type = self.task_type
        model = None
        reference_metric = None
        n_obsv = None
        n_predictors = None
        n_classes = None
        is_multioutput = True

        if task_type == 'regression':
            model = RandomForestRegressor
            reference_metric = "Mean Squared Error (MSE)"

            n_obsv = len(data["y_test"])  # Number of observations
            n_predictors = data["X_test"].shape[1]  # Number of predictors (features)

        elif task_type == 'classification':
            model = RandomForestClassifier
            reference_metric = "F1-Score"

            n_classes = len(np.unique(data["y_train"]))
            is_multioutput = True if isinstance(data["y_train"][0], list) else False

        elif task_type == 'clustering':
            model = KMeans
            reference_metric = "Silhouette Score"
        
        model_and_vars = [model, reference_metric, n_obsv, n_predictors, n_classes, is_multioutput]

        return model_and_vars

    def _get_evaluation_metrics(self):
        """
        To get metrics names and evaluations metrics corrseponding to the selected task type
        """
        task_type = self.task_type

        if task_type == 'regression':
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
            raise ValueError(f"Unsupported task type: {task_type}")

# 4 Set hyperparameter bounds
def hyperparameters_bounds(model, random_state=42):
    # Model Name
    model_name = model.__name__


    if model_name == 'RandomForestRegressor':
        paras_bounds = [
            IntegerVar(lb=10, ub=100, name="n_estimators_paras"),
            StringVar(valid_sets=('squared_error', 'absolute_error', 'friedman_mse', 'poisson'), name="criterion_paras"),
            MixedSetVar(valid_sets=('none', 10, 50, 100), name="max_depth_paras"),
            IntegerVar(lb=2, ub=100, name="min_samples_split_paras"),                     # int in the range [2, inf) or a float in the range (0.0, 1.0]
            IntegerVar(lb=1, ub=5, name="min_samples_leaf_paras"),                      # int in the range [1, inf) or a float in the range (0.0, 1.0)
            # FloatVar(lb=0., ub=0.5, name="min_weight_fraction_leaf_paras"),             # float in the range [0.0, 0.5]
            MixedSetVar(valid_sets=('none', 'sqrt', 'log2', 1, 5, 10, 50, 100), name="max_features_paras"),
            # IntegerVar(lb=2, ub=100, name="max_leaf_nodes_paras"),                      # int in the range [2, inf)
            # FloatVar(lb=1., ub=100., name="min_impurity_decrease_paras"),
            # BoolVar(n_vars=1, name="bootstrap_paras"),                                  # max_sample cannot be set if bootstrap=False. Either switch to bootstrap=True or set max_sample=None
            # BoolVar(n_vars=1, name="oob_score_paras"),                                  # Only available if bootstrap=True
            # MixedSetVar(valid_sets=('none', 10, 50, 100), name="n_jobs_paras"),
            # # IntegerVar(valid_sets=(random_state), name="random_state_paras"),  # Dependant towards bootstrap=True
            # BoolVar(n_vars=1, name="warm_start_paras"),
            # FloatVar(lb=0., ub=100., name="ccp_alpha_paras"),
            # MixedSetVar(valid_sets=('none', 5, 10, 15), name="max_samples_paras"),      # max_sample cannot be set if bootstrap=False. Either switch to bootstrap=True or set max_sample=None
            # # MixedSetVar(valid_sets=('none', -1, 0, -1), name="monotonic_cst_paras"),    # unspported when n_outputs_ > 1 (multioutput regression) or data has missing (NA) values
            # # IntegerVar(lb=0, ub=3, name="verbose_paras"),                             # Irrelevant
        ]

    elif model_name == 'RandomForestClassifier':
        paras_bounds = [
            IntegerVar(lb=10, ub=100, name="n_estimators_paras"),
            StringVar(valid_sets=('gini', 'entropy', 'log_loss'), name="criterion_paras"),
            MixedSetVar(valid_sets=('none', 10, 50, 100), name="max_depth_paras"),
            IntegerVar(lb=2, ub=100, name="min_samples_split_paras"),                     # int in the range [2, inf) or a float in the range (0.0, 1.0]
            IntegerVar(lb=1, ub=5, name="min_samples_leaf_paras"),                      # int in the range [1, inf) or a float in the range (0.0, 1.0)
            # FloatVar(lb=0., ub=0.5, name="min_weight_fraction_leaf_paras"),             # float in the range [0.0, 0.5]
            MixedSetVar(valid_sets=('none', 'sqrt', 'log2', 1, 5, 10, 50, 100), name="max_features_paras"),
            # IntegerVar(lb=2, ub=100, name="max_leaf_nodes_paras"),                      # int in the range [2, inf)
            # FloatVar(lb=1., ub=100., name="min_impurity_decrease_paras"),
            # BoolVar(n_vars=1, name="bootstrap_paras"),                                  # max_sample cannot be set if bootstrap=False. Either switch to bootstrap=True or set max_sample=None
            # BoolVar(n_vars=1, name="oob_score_paras"),                                  # Only available if bootstrap=True
            # MixedSetVar(valid_sets=('none', 10, 50, 100), name="n_jobs_paras"),
            # # IntegerVar(valid_sets=(random_state), name="random_state_paras"),  # Dependant towards bootstrap=True
            # BoolVar(n_vars=1, name="warm_start_paras"),
            # MixedSetVar(valid_sets=('none', 'balanced', 'balanced_subsample'), name="class_weight_paras"),
            # FloatVar(lb=0., ub=100., name="ccp_alpha_paras"),
            # MixedSetVar(valid_sets=('none', 5, 10, 15), name="max_samples_paras"),      # max_sample cannot be set if bootstrap=False. Either switch to bootstrap=True or set max_sample=None
            # MixedSetVar(valid_sets=('none', -1, 0, 1), name="monotonic_cst_paras")      # not supported when n_classes > 2 (multiclass clf), n_outputs_ > 1 (multi-output), or data has missing values
            # # IntegerVar(lb=0, ub=3, name="verbose_paras"),                             # Irrelevant
        ]

    elif model_name == 'KMeans':
        paras_bounds = [
            # FloatVar(lb=1e-5, ub=1e3, name="tol_paras"),
            # StringVar(valid_sets=('linear', 'poly', 'rbf', 'sigmoid'), name="kernel_paras"),
            StringVar(valid_sets=('lloyd', 'elkan'), name="algorithm_paras"),
            IntegerVar(lb=2, ub=10, name="n_clusters_paras"),
            IntegerVar(lb=10, ub=100, name="max_iter_paras"),
            MixedSetVar(valid_sets=('auto', 1, 5, 10, 15), name="n_init_paras"),
            # BoolVar(n_vars=1, name="probability_paras"),
        ]

    return paras_bounds

# 5 Dependencies handling
def dependencies_handling(all_decoded_paras, task_info):
    # Model Name and Default Parameters
    ml_model_name = task_info.model.__name__

    default_params_values = task_info.model().get_params()
    n_classes = task_info.n_classes
    is_multioutput = task_info.is_multioutput

    # Dependency Rules
    dependency_rules = {
        "RandomForestRegressor": {
            "required_keys": {"bootstrap", "max_samples", "oob_score"},
            "handler": lambda params, defaults, **kwargs: {
                "max_samples": None if not params.get("bootstrap", False) else params.get("max_samples", defaults["max_samples"]),
                "oob_score": False if not params.get("bootstrap", False) else params.get("oob_score", defaults["oob_score"]),
            },
        },
        "RandomForestClassifier": {
            "required_keys": {"bootstrap", "max_samples", "oob_score", "class_weight", "warm_start", "monotonic_cst"},
            "handler": lambda params, defaults, n_classes, is_multioutput: {
                "max_samples": None if not params.get("bootstrap", False) else params.get("max_samples", defaults["max_samples"]),
                "oob_score": False if not params.get("bootstrap", False) else params.get("oob_score", defaults["oob_score"]),
                "monotonic_cst": None if n_classes > 2 or is_multioutput else params.get("monotonic_cst", defaults["monotonic_cst"]),
                "warm_start": False if params.get("class_weight") in ("balanced", "balanced_subsample") else params.get("warm_start", defaults["warm_start"]),
            },
        },
    }

    # Apply Rules
    if ml_model_name in dependency_rules:
        rule = dependency_rules[ml_model_name]
        # Ensure required keys are present
        for key in rule["required_keys"]:
            if key not in all_decoded_paras:
                all_decoded_paras[key] = default_params_values[key]

        # Apply handler logic
        handler_result = rule["handler"](
            all_decoded_paras,
            default_params_values,
            n_classes=n_classes,
            is_multioutput=is_multioutput,
        )
        all_decoded_paras.update(handler_result)

    return all_decoded_paras

# 6 Define problem class
class OptimizedProblem(Problem):
    def __init__(
                    self,
                    bounds=None,
                    minmax="max",
                    data=None,
                    task_info=None,
                    **kwargs
                ):
        self.data = data       
        self.task_info = task_info
        self.all_decoded_paras = {}
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, x):
        task_type = self.task_info.task_type
        all_decoded_paras = self.all_decoded_paras
        original_paras = {}

        x_decoded = self.decode_solution(x)

        paras_bounds_names = [] # List containing names of the hyperparameters
        for i, _ in enumerate(self.bounds):
            paras_bounds_names.append(self.bounds[i].name)  # Store each of parameter name (w/ "")

        for paras_name in paras_bounds_names:
            original_paras[paras_name] = x_decoded[paras_name]
            all_decoded_paras[paras_name[:-6]] = None if original_paras[paras_name] == 'none' else original_paras[paras_name]

        # Decoded paras (dict) after handling dependecies
        all_decoded_paras = dependencies_handling(all_decoded_paras, self.task_info)

        # Defining the model and assigning hyperparameters
        if 'random_state' in self.task_info.model().get_params():
            all_decoded_paras['random_state'] = 42 
        ml_model = self.task_info.model(**all_decoded_paras)
        
        # Supervised tasks
        if task_type in ('regression', 'classification'):
            ml_model.fit(self.data["X_train"], self.data["y_train"])
            y_predict = ml_model.predict(self.data["X_test"])

            # MSE for Regression
            if task_type == 'regression':
                return mean_squared_error(self.data["y_test"], y_predict)

            # F1-Score for Classification
            elif task_type == 'classification':
                return f1_score(self.data["y_test"], y_predict, average='macro')

        # Unsupervised tasks (Clustering)
        elif task_type == 'clustering':
            ml_model.fit_predict(self.data["X"])
            labels = ml_model.fit_predict(self.data["X"])
            
            # Silhouette Score for Clustering
            return silhouette_score(self.data["X"], labels)

# 7 Re-fit and re-predict using best parameters
# 7.2 Re-fit and re-predict
def optimized_fit_predict(optimizer,
                            data,
                            task_type,
                            task_info,
                            ):
    # Get best parameters after optimization
    best_paras = optimizer.problem.decode_solution(optimizer.g_best.solution)
    
    best_paras_decoded = {}
    for key, value in best_paras.items():
        best_paras_decoded[key[:-6]] = None if value == 'none' else value

    # Apply dependency handling (ensure this doesn't overwrite the decoded values)
    best_paras_decoded = dependencies_handling(best_paras_decoded, task_info)

    if 'random_state' in task_info.model().get_params():
        best_paras_decoded['random_state'] = 42 
    ml_model = task_info.model(**best_paras_decoded)

    if task_type in ('regression', 'classification'):
        ml_model.fit(data["X_train"], data["y_train"])
        y_predict = ml_model.predict(data["X_test"])

        if task_type == 'regression':
            metrics = task_info.eval_metrics(data["y_test"], y_predict, task_info.n_obsv, task_info.n_predictors)

        elif task_type == 'classification':
            metrics = task_info.eval_metrics(data["y_test"], y_predict, task_info.n_classes)

    elif task_type == 'clustering':
        ml_model.fit_predict(data["X"])
        labels = ml_model.fit_predict(data["X"])
        metrics = task_info.eval_metrics(data["X"], labels)
    
    return [ml_model, metrics]

# --- End of evaluation.py ---

# --- Start of visualization.py ---
# import streamlit as st

# import os
# import time
# import datetime
# import joblib

# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# from scipy.stats import shapiro

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from sklearn.cluster import KMeans
# from sklearn.inspection import permutation_importance


# from sklearn.metrics import (
#     mean_squared_error,
#     mean_absolute_error,
#     mean_absolute_percentage_error,
#     r2_score,
#     explained_variance_score,

#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,

#     silhouette_score,
#     davies_bouldin_score,
#     calinski_harabasz_score,
# )

# from mealpy import FloatVar, StringVar, IntegerVar, BoolVar, MixedSetVar, Problem, ACOR, GA, PSO, SA

# Data Preprocessing

"""
Variables needed:
- None

Returns:
- None (images are stored in the './feature_importance_plots')
"""

# 1 Compute feature importance
# 2 Visualze feature importance

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
def feature_importance_visualization(data, task_type, is_cloud_env, optimizer_name=None, bucket_name=None):
    # Define the folder path
    folder_path = 'feature_importance_plots'

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

        save_plot(plot_title, folder_path, is_cloud_env, bucket_name)
    else:
        plot_title = f"Feature Importance ({optimizer_name})"
        # Sort features by importance
        data = data.sort_values(by='Importance', ascending=False)

        # Visualize the feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=data)
        plt.title(plot_title, fontsize=16)
        plt.xlabel("Importance", fontsize=12)
        plt.ylabel("Features", fontsize=12)

        save_plot(plot_title, folder_path, is_cloud_env, bucket_name)

        # plt.tight_layout()
        # plt.show() 

# # Visualize for all metaopt
# def final_visualization(result_df, task_type, data_dict, is_cloud_env, bucket_name):
#     optimizers_names = result_df.iloc[:,0]
#     ml_models = result_df.iloc[:,2]

#     # Iterate through each model
#     for optimizer_name, best_ml_model in zip(optimizers_names, ml_models):

#         # Compute the feature importance
#         feature_importance_data = compute_feature_importance(best_ml_model, data_dict)
        
#         # Generate the visualization
#         feature_importance_visualization(feature_importance_data, task_type, is_cloud_env, optimizer_name, bucket_name)
# --- End of visualization.py ---

# --- Start of pipeline.py ---
class MetaheuristicPipeline:
    def __init__(self, filepath, task_type, label):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier()
        self.encoders = {}
    
    def load_data(self, filepath):
        # Load dataset
        return pd.read_csv(filepath)

    # Preprocessing
    def preprocessing(self, data, task_type, label=None):

        # If there's no label input
        if label is None:
            label = data.columns[-1]

        data = data_cleaning(data, task_type, label)
        data = data_transformation(self, data, task_type, label)
        
        return data

    # Generate EDA Plots
    def eda_visualization(self, data, task_type, is_cloud_env, label=None, bucket_name=None, columns_to_plot=None):
        """
        Perform Exploratory Data Analysis with plots and statistics.

        Parameters
        ----------
        data : Pandas DataFrame or list
            DataFrame or list containing train/test splits [X_train, X_test, y_train, y_test].
        task_type : str
            The task type ('regression', 'classification').
        is_cloud_env : bool
            Indicates if the environment is cloud-based.
        label : str, optional
            Label column name for classification or regression tasks.
        bucket_name : str, optional
            Cloud storage bucket name (for cloud environments).
        columns_to_plot : list, optional
            List of column names to include in visualizations.
        """

        if task_type in ('regression', 'classification') and isinstance(data, list) and len(data) == 4:
            # Combine train and test datasets
            X_combined = pd.concat([data[0], data[1]], axis=0).reset_index(drop=True)
            y_combined = pd.concat([pd.Series(data[2], name=label), pd.Series(data[3], name=label)], axis=0).reset_index(drop=True)
            data = pd.concat([X_combined, y_combined], axis=1)

        # Filter columns for plotting
        if columns_to_plot:
            data = data[columns_to_plot]

        folder_path = 'eda_plots'
        if not is_cloud_env:
            os.makedirs(folder_path, exist_ok=True)

        # Pairplot
        pairplot_title = f"Pairwise Relationship Plot ({task_type})"
        sns.pairplot(data.select_dtypes(include=[np.number]))
        plt.gcf().suptitle(pairplot_title, y=1.02)
        save_plot(pairplot_title + ".png", folder_path, is_cloud_env, bucket_name)

        # Boxplot
        for column in data.select_dtypes(include=[np.number]).columns:
            boxplot_title = f"Boxplot for {column} ({task_type})"
            sns.boxplot(data[column])
            plt.title(boxplot_title)
            save_plot(boxplot_title + ".png", folder_path, is_cloud_env, bucket_name)

        # Correlation Matrix
        corr_mtx_title = f"Correlation Matrix ({task_type})"
        matrix = data.select_dtypes(include=[np.number]).corr()
        sns.heatmap(matrix, cmap="Blues", annot=True, fmt=".2f")
        plt.title(corr_mtx_title)
        save_plot(corr_mtx_title + ".png", folder_path, is_cloud_env, bucket_name)

    # Evaluation
    def evaluate(self, data, task_type, bucket_name_model=None, ant_colony=None, genetic_algorithm=None, particle_swarm=None, simulated_annealing=None):
        # Assign data into specified cases
        data_dict = create_data_dict(data, task_type)

        # Model and necessary variable(s)
        task_info = TaskInfo(data=data_dict, task_type=task_type)

        # Setting the min-max value
        minimize_metrics = ["Mean Squared Error (MSE)",
                            "Root Mean Squared Error (RMSE)",
                            "Mean Absolute Error (MAE)",
                            "Mean Absolute Percentage Error (MAPE)",
                            "Davies-Bouldin Index",
                            ]
        minmax_val = "min" if task_info.reference_metric in minimize_metrics else "max"

        # Getting hyperparameter bounds and names
        paras_bounds = hyperparameters_bounds(task_info.model, random_state=42)

        epoch = 10
        pop_size = 10

        # Assigning Metaheursitic Optimizer
        optimizers = [
            ACOR.OriginalACOR(epoch=epoch, pop_size=pop_size, sample_count = 25, intent_factor = 0.5, zeta = 1.0) if ant_colony is None else ant_colony,
            GA.BaseGA(epoch=epoch, pop_size=pop_size, pc=0.9, pm=0.05, selection="tournament", k_way=0.4, crossover="multi_points", mutation="swap") if genetic_algorithm is None else genetic_algorithm,
            PSO.OriginalPSO(epoch=epoch, pop_size=pop_size, c1=2.05, c2=2.05, w=0.4) if particle_swarm is None else particle_swarm,
            SA.OriginalSA(epoch=epoch, pop_size=pop_size, temp_init=100, step_size=0.1) if simulated_annealing is None else simulated_annealing
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
                                        task_info=task_info,
                                        )

            # Time monitoring and optimization process
            start = time.perf_counter()
            optimizer.solve(problem)
            end = time.perf_counter() - start
            
            best_ml_model, best_metrics_opt = optimized_fit_predict(optimizer = optimizer,
                                                                    data = data_dict,
                                                                    task_type = task_type,
                                                                    task_info = task_info,
                                                                    )

            metaopt_name.append(optimizer.__class__.__name__)

            metaopt_object.append(optimizer)
            ml_models.append(best_ml_model)
            best_metrics.append(best_metrics_opt)
            time_taken.append(end)

            print(f"Best agent: {optimizer.g_best}")
            print(f"Best solution: {optimizer.g_best.solution}")
            print(f"Best {task_info.reference_metric}: {optimizer.g_best.target.fitness}")
            print(f"Best parameters: {optimizer.problem.decode_solution(optimizer.g_best.solution)}\n")        

        # Final result
        result_df = pd.DataFrame ({
            "Metaheuristic Optimizer (Name)" : metaopt_name,
            "Metaheuristic Optimizer (Object)" : metaopt_object,
            "Machine Learning Model (object)" : ml_models,
            **{metric: values for metric, values in zip(task_info.metrics_names, zip(*best_metrics))},
            "Time taken (s)" : time_taken,
        })

        # Save the trained model
        ascending = None
        if minmax_val == "max":
            ascending = False
        else:
            ascending = True

        best_ml_model = result_df.sort_values(by=task_info.reference_metric, ascending=ascending).iloc[0,2]
        best_ml_model_name = f'Best_{best_ml_model.__class__.__name__}.pkl'
        joblib.dump(best_ml_model, best_ml_model_name)

        try:
            # Initialize client
            client = storage.Client()
            bucket = client.bucket(bucket_name_model)

            # Blob (file in the bucket)
            blob = bucket.blob(best_ml_model_name)

            # # Upload Pickle object (from memory)
            # blob.upload_from_string(best_ml_model_name)

            # OR Upload Pickle file (from disk)
            blob.upload_from_filename(best_ml_model_name)
        except Exception as e:
            return f"Error during saving model: {e}"

        return result_df, best_ml_model, data_dict

    # Visualize for all metaopt
    def final_visualization(self, result_df, task_type, data_dict, is_cloud_env, bucket_name):
        optimizers_names = result_df.iloc[:,0]
        ml_models = result_df.iloc[:,2]

        # Iterate through each model
        for optimizer_name, best_ml_model in zip(optimizers_names, ml_models):

            # Compute the feature importance
            feature_importance_data = compute_feature_importance(best_ml_model, data_dict)
            
            # Generate the visualization
            feature_importance_visualization(feature_importance_data, task_type, is_cloud_env, optimizer_name, bucket_name)
        
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


# --- End of pipeline.py ---