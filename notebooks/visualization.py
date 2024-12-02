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

# Data Preprocessing

"""
Variables needed:
- None

Returns:
- None (images are stored in the '../feature_importance_plots/')
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

# 2 Visualze feature importance for Dashboard
def feature_importance_visualization(data, task_type, optimizer_name=None):
    
    # Define the folder path
    folder_path = 'feature_importance_plots'
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
        st.pyplot(plt)
        # plt.show()
    else:
        plot_title = f"Feature Importance ({optimizer_name})"
        # Sort features by importance
        data = data.sort_values(by='Importance', ascending=False)

        # Visualize the feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=data, palette='viridis')
        plt.title(plot_title, fontsize=16)
        plt.xlabel("Importance", fontsize=12)
        plt.ylabel("Features", fontsize=12)
        plt.tight_layout()
        st.pyplot(plt)
        # plt.show()
    
    # save_plot(folder_path, plot_title)

# # 2 Visualze feature importance
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
#         plt.show()
#     else:
#         plot_title = f"Feature Importance ({optimizer_name})"
#         # Sort features by importance
#         data = data.sort_values(by='Importance', ascending=False)

#         # Visualize the feature importance
#         plt.figure(figsize=(10, 6))
#         sns.barplot(x='Importance', y='Feature', data=data, hue=y, legend=False)#, palette='viridis')
#         plt.title(plot_title, fontsize=16)
#         plt.xlabel("Importance", fontsize=12)
#         plt.ylabel("Features", fontsize=12)
#         plt.tight_layout()
#         plt.show()
    
#     save_plot(folder_path, plot_title)

# # Visualize for all metaopt
# for result_df, data_dict, task_type in zip(result_df_list, data_dict_list, task_type_list):

# # Optimizers names
# optimizers_names = result_df[result_df.columns[0]]

# # Machine Learning Models
# ml_models = result_df.iloc[:,2]

# # Iterate through each model
# for optimizer_name, best_ml_model in zip(optimizers_names, ml_models):

#     # Compute the feature importance
#     feature_importance_data = compute_feature_importance(best_ml_model, data_dict)
    
#     # Generate the visualization
#     feature_importance_visualization(feature_importance_data, task_type, optimizer_name)