from data_analysis import Analysis, get_analysis
from data_ingestion import load_yaml_file, get_data_info
from data_pipeline import *
from pprint import pprint
from sklearn.model_selection import train_test_split
import json
import polars as pl
from polars import selectors as cs
import numpy as np


# ------------------------------------------------------------------- #
if __name__ == "__main__":
    # Load data and schema
    df, schema = load_yaml_file(filepath="data.yaml")
    df_schema = get_data_info(df)
    
    # Data Cleaning
    drop_cols = schema.get('drop_columns', [])
    df = df.drop(drop_cols)
    df = df.with_columns(
            cs.numeric().fill_null(0),
            cs.string().fill_null("missing_value"),
            cs.temporal().fill_null(pl.lit("1970-01-01").str.to_datetime())
        )
    print("Data cleaning done...")
    
    # Get user's input for type of task to do
    print("1. Regression \t 2. Classification \t 3. Clustering")
    task_type = ''
    while task_type not in ['1', '2', '3']:
        task_type = input("What task do you want to perform? Enter the corresponding number: ")
    
    # Analyze the dataset using Gemini 2.5 Flash-lite 
    analysis = get_analysis(df.head(), df_schema, task_type)
    json_analysis = json.loads(analysis.model_dump_json())
    print(json_analysis)
    
    # Create Transformation Pipeline from Analysis object
    pipeline = create_transformation_pipeline(analysis)
    print(pipeline)
    print("Data Transformation...")
    df_transformed = pipeline.fit_transform(X=df)
    
    # Create Model Pipeline for training on the dataset
    model_pipeline = create_model_pipeline(analysis, which_task=int(task_type))
    print(model_pipeline)
    if task_type in ['1', '2']:
        target_col = schema['target_column']
        X = df.drop(target_col)
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print("Model training...")
        model_pipeline.fit(X_train, y_train)
        params = model_pipeline.get_params()
        print(params)
        
        score = model_pipeline.score(X_test, y_test)
        print(f"Model Score: {score}")
    else:
        model_pipeline.fit(df_transformed)
        print("Clustering completed.")