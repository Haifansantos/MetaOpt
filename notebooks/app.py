import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from mealpy.evolutionary_based.GA import BaseGA  # Adjust based on actual usage
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from itertools import combinations

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

### Import modules
import load_data
import preprocessing
import eda
import evaluation
import visualization

from evaluation import create_data_dict
from visualization import feature_importance_visualization

# Define your data_transformation, feature_target_split, feature_target_encoding, handle_model_dependencies, decode_best_paras, evaluate functions here
# Ensure all necessary functions are defined

def main():
    st.title("Metaheuristic Hyperparameter Optimization Dashboard")
    
    # Sidebar for user inputs
    st.sidebar.header("User Inputs")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            data = load_data.load_data(uploaded_file) # Data is loaded directly
            st.write("## Dataset Preview")
            st.dataframe(data.head())
        except Exception as e:
            st.error(f"Error reading the CSV file: {e}")
            return
        
        # Task type selection
        task_type = st.sidebar.selectbox("Select Task Type", ("Regression", "Classification", "Clustering"))
        task_type = task_type.lower()

        if task_type != 'clustering':
            colnames = data.columns
            colnames = list(colnames)
            colnames.append('None')

            # Optional: Allow user to specify label column
            label = st.sidebar.selectbox("Select Label Column (select \"None\" for last column; default to last column)", colnames, index=colnames.index(colnames[-2]))
            if label == 'None':
                label = None
        else:
            label = None

        # Preprocessing
        with st.spinner("Preprocessing data..."):
            processed_data = preprocessing.preprocessing(data, task_type, label)
            if processed_data is None:
                st.error("Data transformation failed.")
                return
            st.success("Data preprocessing completed.")
        
        
        st.write("## Preprocessed Data")
        if task_type != 'clustering' and isinstance(processed_data, list) and len(processed_data) == 4:
            # Combine train and test features
            X_combined = pd.concat([processed_data[0], processed_data[1]], axis=0).reset_index(drop=True)

            # Combine train and test labels
            y_train_series = pd.Series(processed_data[2], name=label).reset_index(drop=True)
            y_test_series = pd.Series(processed_data[3], name=label).reset_index(drop=True)
            y_combined = pd.concat([y_train_series, y_test_series], axis=0).reset_index(drop=True)

            # Combine features and labels into a single DataFrame
            processed_data_display = pd.concat([X_combined, y_combined], axis=1)
        else:
            processed_data_display = processed_data
        
        st.dataframe(processed_data_display.head())
        
        # # Display the EDA visualizations
        # if st.button("Generate EDA Plots"):
        #     eda.eda_visualization(processed_data, task_type, label)
        
        # Exploratory Data Analysis
        st.write('## Exploratory Data Analysis (EDA)')
        eda.eda_visualization(processed_data, task_type, label)

        st.write('## Optimization')
        # Optimization Button
        if st.button("Start Optimization"):
            with st.spinner("Running metaheuristic optimization..."):
                try:
                    result_df, data_dict, reference_metric = evaluation.evaluate(data=processed_data, task_type=task_type)
                    st.success("Optimization completed.")
                    
                    # Display results
                    st.write("### Optimization Results")
                    st.dataframe(result_df.drop(result_df.columns[[1, 2]], axis=1))
                    
                    st.write("### Best Model")

                    ascending = True if reference_metric in ["Mean Squared Error (MSE)",
                                                            "Root Mean Squared Error (RMSE)",
                                                            "Mean Absolute Error (MAE)",
                                                            "Mean Absolute Percentage Error (MAPE)",
                                                            "Davies-Bouldin Index",
                                                            ] else False
                    best_ml_model = result_df.sort_values(by=reference_metric, ascending=ascending).iloc[0,2]

                    st.text(best_ml_model)
                    
                    # # Display metrics
                    # st.write("## Performance Metrics")
                    # st.write(data_dict.get('metrics', {}))
                    
                    # Feature Importance Visualization
                    st.write('### Feature Importance')

                    # Optimizers names
                    optimizers_names = result_df[result_df.columns[0]]

                    # Machine Learning Models
                    ml_models = result_df.iloc[:,2]

                    # Iterate through each model
                    for optimizer_name, best_ml_model in zip(optimizers_names, ml_models):

                        # Compute the feature importance
                        feature_importance_data = visualization.compute_feature_importance(best_ml_model, data_dict, label)
                        
                        st.write(f'#### {optimizer_name}')
                        # Generate the visualization
                        feature_importance_visualization(feature_importance_data, task_type, optimizer_name)


                    # if hasattr(best_ml_model, "feature_importances_"):
                    #     feature_importance = best_ml_model.feature_importances_
                    #     feature_names = processed_data[0].columns  # Assuming X_train is data[0]
                    #     fi_df = pd.DataFrame({
                    #         'Feature': feature_names,
                    #         'Importance': feature_importance
                    #     }).sort_values(by='Importance', ascending=False)
                        
                    #     st.write("## Feature Importances")
                    #     fig, ax = plt.subplots(figsize=(10, 6))
                    #     sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis', ax=ax)
                    #     plt.title("Feature Importance")
                    #     st.pyplot(fig)
                    
                        # Additional Visualizations for Clustering
                        if task_type.capitalize() == 'Clustering' and hasattr(best_ml_model, "labels_"):
                            labels = best_ml_model.labels_
                            X = processed_data

                            # Ensure that X has more than one column
                            if X.shape[1] > 1:
                                
                                for i in range(1, X.shape[1] + 1):

                                    if i in (2, 3):
                                        column_combinations = combinations(X.columns[:-1], i)
                                        
                                        for comb in column_combinations:
                                            # if i == 2:
                                            #     fig, ax = plt.subplots(figsize=(10, 6))
                                                
                                            #     # Create custom scatterplots for pairs of features (e.g., columns 0 and 1)
                                            #     sns.scatterplot(x=X[comb[0]], y=X[comb[1]], hue=labels, palette='viridis', ax=ax)
                                            #     plt.title(f"Cluster Assignments ({comb[0]} vs {comb[1]})")
                                            #     st.pyplot(fig)
                                            if i == 3:
                                                # Create a new DataFrame with the cluster labels
                                                X['Cluster'] = labels
                                                
                                                # Create 3D scatter plot with Plotly
                                                fig = px.scatter_3d(X, x=X[comb[0]], y=X[comb[1]], z=X[comb[2]], color='Cluster', 
                                                                    title=f"Cluster Assignments (3D) - {comb[0]} vs {comb[1]} vs {comb[2]}", 
                                                                    color_continuous_scale='Viridis')
                                                
                                                # Show the plot in Streamlit
                                                st.plotly_chart(fig)
                    
                except Exception as e:
                    st.error(f"Error during optimization: {e}")

if __name__ == "__main__":
    main()


