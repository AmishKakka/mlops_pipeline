from data_analysis import Analysis, get_analysis
from data_ingestion import load_yaml_file, get_data_info
from data_pipeline import *
from pprint import pprint
from sklearn.model_selection import train_test_split
import json
import polars as pl
from polars import selectors as cs
import numpy as np

# Importing Models from sklearn
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering


REGRESSOR_MAP = {
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "SVR": SVR,
}

CLASSIFIER_MAP = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "SVC": SVC,
    "KNeighborsClassifier": KNeighborsClassifier,
}

CLUSTERER_MAP = {
    "KMeans": KMeans,
    "DBSCAN": DBSCAN,
    "AgglomerativeClustering": AgglomerativeClustering,
}


# ------------------------------------------------------------------- #
if __name__ == "__main__":
    # Load data and schema
    df, schema = load_yaml_file(filepath="data.yaml")
    df_schema = get_data_info(df)
    
    # Get user's input for type of task to do
    print("1. Regression \t 2. Classification \t 3. Clustering")
    task_type = ''
    while task_type not in [1, 2, 3]:
        task_type = int(input("What task do you want to perform? Enter the corresponding number: "))
    
    # Analyze the dataset using Gemini 2.5 Flash-lite 
    analysis = get_analysis(df.head(), df_schema, task_type)
    json_analysis = json.loads(analysis.model_dump_json())
    print(json_analysis)
    
    # Create Transformation Pipeline from Analysis object
    df_transformed = transformation_pipeline(analysis, df)
    print(df_transformed)


    # -------------------------------------------------------------------------- #
    # Regression task
    if task_type == 1:
        # Splitting the data into 'train' and 'test'
        target_col = schema['target_column']
        X = df_transformed.drop(target_col)
        y = df_transformed[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print("Model training...")
        # Going through the list of suggested models...
        for model_suggested in analysis.suggested_regression_models:
            # Getting the model name and initializing it...
            model_cls = REGRESSOR_MAP[model_suggested.model_name]
            model = model_cls(**model_suggested.hyperparameters)
            
            # Model training...
            trained_model = model.fit(X_train, y_train)
            pprint(trained_model.get_params())

            # Getting R^2 score for Regression model...
            score = trained_model.score(X_test, y_test)
            print(f"{model_suggested.model_name} Model Score: {score:.3f}")
    
    # Classification task
    elif task_type == 2:
        # Splitting the data into 'train' and 'test'
        target_col = schema['target_column']
        X = df_transformed.drop(target_col)
        y = df_transformed[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print("Model training...")
        # Going through the list of suggested models...
        for model_suggested in analysis.suggested_classification_models:
            # Getting the model name and initializing it...
            model_cls = CLASSIFIER_MAP[model_suggested.model_name]
            model = model_cls(**model_suggested.hyperparameters)
            
            # Model training...
            trained_model = model.fit(X_train, y_train)
            pprint(trained_model.get_params())

            # Getting the accuracy for the Classification model...
            score = trained_model.score(X_test, y_test)
            print(f"{model_suggested.model_name} Model Score: {score:.3f}")
    
    # Clustering task
    else:
        print("Model training...")
        # Going through the list of suggested models...
        for model_suggested in analysis.suggested_clustering_models:
            # Getting the model name and initializing it...
            model_cls = CLUSTERER_MAP[model_suggested.model_name]
            model = model_cls(**model_suggested.hyperparameters)
            
            # Model training...
            model.fit(df_transformed)
            pprint(model.get_params())
            
            # Getting the accuracy for the Clustering model...
            try:
                score = model.score(df_transformed)
                print(f"{model_suggested.model_name} Model Score: {score:.3f}")
            except:
                print(f"{model_suggested.model_name} has no score evaluation function.")