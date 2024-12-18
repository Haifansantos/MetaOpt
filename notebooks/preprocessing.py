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

# Preprocessing
def preprocessing(data, task_type, label=None):

    # If there's no label input
    if label is None:
        label = data.columns[-1]

    data = data_cleaning(data, task_type, label)
    data = data_transformation(data, task_type, label)
    
    return data