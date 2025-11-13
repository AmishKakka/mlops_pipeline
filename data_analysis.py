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
        "LabelEncoder",
        "FunctionTransformer"
    ] = Field(
        description="The sklearn class name of the transformer to use.")
    parameters: Optional[Dict[str, Any]] = Field(
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
    hyperparameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="A dictionary of hyperparameters for the model.")

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
    hyperparameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="A dictionary of hyperparameters for the model.")

class Clusterer(BaseModel):
    """A structured clustering model."""
    model_name: Literal[
        "KMeans",
        "DBSCAN",
        "AgglomerativeClustering"
    ] = Field(description="The sklearn clustering class name.")
    hyperparameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="A dictionary of hyperparameters for the model.")

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
    

def get_analysis(schema):
    client = genai.Client(api_key=os.getenv("API_Key"))
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=f"You are an expert data scientist. Please analyze the dataset with schema: {schema} and provide a complete machine learning plan. \
                What transformations should i apply, what kind of tasks can be done, and which algorithms along with their parameters can be used? \
                Remember you don't necessarily need to mention models and transformations. Mention if they are required.",
        config={
            "response_mime_type": "application/json",
            "response_json_schema": Analysis.model_json_schema(),
        }
    )
    print(response)