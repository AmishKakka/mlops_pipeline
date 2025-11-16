from google import genai
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
import os


class Transformation(BaseModel):
    """A structured data transformation step."""
    columns: List[str] = Field(
        description="List of column names this transformation applies to.")
    transformer: Literal[
        "SimpleImputer",
        "StandardScaler",
        "MinMaxScaler",
        "RobustScaler",
        "OneHotEncoder",
        "OrdinalEncoder",
        "LabelEncoder"
    ] = Field(description="The sklearn class name of the transformer to use.")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="A dictionary of parameters for the transformer, e.g., {'strategy': 'median'} for SimpleImputer.")

class Classifier(BaseModel):
    """A structured classification model."""
    model_name: Literal[
        "LogisticRegression",
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "SVC",
        "KNeighborsClassifier"
    ] = Field(description="The sklearn classifier class name.")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="A dictionary of hyperparameters for the model.")

class Regressor(BaseModel):
    """A structured regression model."""
    model_name: Literal[
        "LinearRegression",
        "Ridge",
        "Lasso",
        "RandomForestRegressor",
        "GradientBoostingRegressor",
        "SVR"
    ] = Field(description="The sklearn regressor class name.")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="A dictionary of hyperparameters for the model.")

class Clusterer(BaseModel):
    """A structured clustering model."""
    model_name: Literal[
        "KMeans",
        "DBSCAN",
        "AgglomerativeClustering"
    ] = Field(description="The sklearn clustering class name.")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="A dictionary of hyperparameters for the model.")

class Analysis(BaseModel):
    """A complete and structured analysis plan."""
    overview: str = Field(
        description="A brief overview of the dataset, like how many columns and rows it has, what kind of data it contains, \
            what data types are present?")
    transformations: List[Transformation] = Field(
        description="A structured list of data transformations to perform.")
    suggested_classification_models: Optional[List[Classifier]] = Field(
        default=None, 
        description="A list of suggested classification models, if classification is a relevant task.")
    suggested_regression_models: Optional[List[Regressor]] = Field(
        default=None, 
        description="A list of suggested regression models, if regression is a relevant task.")
    suggested_clustering_models: Optional[List[Clusterer]] = Field(
        default=None, 
        description="A list of suggested clustering models, if clustering is a relevant task.")
    

def get_analysis(df_head, schema, task_type):
    task = {1: "Regression", 2: "Classification", 3: "Clustering"}
    client = genai.Client(api_key=os.getenv("API_Key"))
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=f"You are an expert data scientist. Please analyze the dataset containing sample data - {df_head.to_dicts()} and schema: {schema}. \
                I want to perform {task[int(task_type)]} task on this dataset. \
                **TRANSFORMATION RULES:** \
                    1. Suggest transformations from this *exact* list: ['SimpleImputer', 'StandardScaler', 'MinMaxScaler', 'RobustScaler', 'OneHotEncoder', 'OrdinalEncoder']. \
                    2. **CRITICAL:** NEVER suggest 'LabelEncoder'. It is only for target variables, not features. Use 'OrdinalEncoder' for categorical features instead. \
                    3. **CRITICAL:** If you suggest 'OneHotEncoder', you MUST include the parameter 'sparse_output=False' in its parameters dictionary. \
                \
                **MODEL RULES:** \
                    1. Which task is most appropriate: 'classification', 'regression', or 'clustering'? \
                    2. Based *only* on that task, suggest models.\
                    3. If 'classification', suggest from: ['LogisticRegression', 'RandomForestClassifier', 'GradientBoostingClassifier', 'SVC', 'KNeighborsClassifier']. \
                    4. If 'regression', suggest from: ['LinearRegression', 'Ridge', 'Lasso', 'RandomForestRegressor', 'GradientBoostingRegressor', 'SVR']. \
                    5. If 'clustering', suggest from: ['KMeans', 'DBSCAN', 'AgglomerativeClustering']. \
                ",
        config={
            "response_mime_type": "application/json",
            "response_json_schema": Analysis.model_json_schema(),
        }
    )
    analysis = Analysis.model_validate_json(response.text)
    return analysis