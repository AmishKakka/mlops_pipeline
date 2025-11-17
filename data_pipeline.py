from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


TRANSFORMER_MAP = {
    "SimpleImputer": SimpleImputer,
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
    "RobustScaler": RobustScaler,
    "OrdinalEncoder": OrdinalEncoder,
    "LabelEncoder": LabelEncoder
}


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