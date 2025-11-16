from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


TRANSFORMER_MAP = {
    "SimpleImputer": SimpleImputer,
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
    "RobustScaler": RobustScaler,
    "OneHotEncoder": OneHotEncoder,
    "OrdinalEncoder": OrdinalEncoder,
    "LabelEncoder": LabelEncoder
}

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


def create_model_pipeline(analysis, which_task: int):
    """Create a sklearn Pipeline with model from an Analysis object based on the task."""
    steps = []
    if which_task == 1:
        model_info = analysis.suggested_regression_models[1]
        model_cls = REGRESSOR_MAP[model_info.model_name]
    elif which_task == 2:
        model_info = analysis.suggested_classification_models[0]
        model_cls = CLASSIFIER_MAP[model_info.model_name]
    elif which_task == 3:
        model_info = analysis.suggested_clustering_models[0]
        model_cls = CLUSTERER_MAP[model_info.model_name]
    else:
        raise ValueError("Unsupported model type")
    
    model = model_cls(**model_info.hyperparameters)
    steps.append(('model', model))
    pipeline = Pipeline(steps=steps, verbose=True)
    print("Model Pipeline created...")
    return pipeline

def create_transformation_pipeline(analysis):
    """Create a sklearn Pipeline from an Analysis object."""
    steps = []
    
    # Add transformations
    if analysis.transformations:
        transformers = []
        for transform in analysis.transformations:
            transformer_cls = TRANSFORMER_MAP[transform.transformer]
            transformer = transformer_cls(**transform.parameters)
            name = f"{transform.transformer.lower()}_{'_'.join(transform.columns)}"            
            transformers.append((name, transformer, transform.columns))
        col_transformer = ColumnTransformer(transformers=transformers, remainder='passthrough')
        col_transformer.set_output(transform="pandas")
        steps.append(('preprocessor', col_transformer))
    pipeline = Pipeline(steps=steps, verbose=True)
    print("Transformation Pipeline created...")
    return pipeline

