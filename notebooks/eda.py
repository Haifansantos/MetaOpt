import streamlit as st

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

# Exploratory Data Analysis (EDA)

"""
Variables needed:
- None

Returns:
- None (images are stored within '../eda_plots/')
"""

# Save plot func
# eda_visualization

# Save plot
def save_plot(folder_path, plot_title):
    file_path = os.path.join(folder_path, plot_title)
    
    if os.path.exists(file_path):
        pass
    else:
        plt.savefig(file_path, dpi=300, bbox_inches='tight')

# EDA for dashboard
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

    # Plotting the Pairwise relationship in the dataset
    pairplot_title = "Pairwise relationship plot"
    sns.pairplot(data)
    plt.gcf().suptitle(pairplot_title, y=1.02)
    st.pyplot(plt)
    plt.figure()

    # Plotting the Boxplot for all the columns in the dataset
    for column_name in data.columns:
        boxplot_title = f"Boxplot for the {column_name} column"
        sns.boxplot(data[column_name])
        plt.title(boxplot_title)
        st.pyplot(plt)
        plt.figure()

    # Displaying correlation matrix of the features in the dataset
    corr_mtx_title = "Correlation Matrix"
    matrix = data.corr()
    sns.heatmap(matrix, cmap="Blues", annot=True)
    plt.title(corr_mtx_title)
    st.pyplot(plt)

# # eda visualization
# def eda_visualization(data, task_type, label=None):
#     """
#     Displaying the Dataset Information, Dataset description, and
#     Plotting a pairplot, Boxplot, and Heatmap of the correlation matrix of the features

#     Parameters
#     ------
#     data  : Pandas DataFrame
#         DataFrame from which the Info, Desc, and Pairplot is retrieved 
#     """

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

#     # Define the folder path
#     folder_path = 'eda_plots'
#     os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist
    
#     # Plotting the Pairwise relationship in the dataset
#     pairplot_title = f"Pariwise relationship plot ({task_type})"
    
#     sns.pairplot(data)
#     plt.gcf().suptitle(pairplot_title, y=1.02)
    
#     save_plot(folder_path, pairplot_title)
#     plt.figure()

#     # Plotting the Boxplot for all the columns in the dataset
#     for column_name in data.columns:
#         boxplot_title = f"Boxplot for the {column_name} column ({task_type})"

#         sns.boxplot(data[column_name])
#         plt.title(boxplot_title)
        
#         save_plot(folder_path, boxplot_title)
#         plt.figure()

#     # Displaying correlation matrix of the features in the dataset
#     corr_mtx_title = f"Correlation Matrix ({task_type})"
#     matrix = data.corr()

#     sns.heatmap(matrix, cmap="Blues", annot=True)
#     plt.title(corr_mtx_title)
    
#     save_plot(folder_path, corr_mtx_title)
#     plt.show()
