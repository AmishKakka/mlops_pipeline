import polars as pl


def SimpleImputer(df: pl.DataFrame, col_names: list[str], strategy: str = 'mean', fill_value = None) -> pl.DataFrame:
    """
    Imputes missing values in specified columns.
    
    Strategies: 'mean', 'median', 'mode', 'constant'
    'fill_value' is required if strategy='constant'
    """
    if strategy == 'forward_fill' or strategy == 'ffill':
        return df.with_columns(
            pl.col(col_names).forward_fill()
        )
    
    if strategy == 'backward_fill' or strategy == 'bfill':
        return df.with_columns(
            pl.col(col_names).backward_fill()
        )
    
    if strategy == 'mean':
        fill_expr = pl.col(col_names).mean()
    elif strategy == 'median':
        fill_expr = pl.col(col_names).median()
    elif strategy == 'mode':
        fill_expr = pl.col(col_names).mode().first()
    elif strategy == 'constant':
        if fill_value is None:
            raise ValueError("fill_value must be provided when strategy is 'constant'")
        fill_expr = fill_value
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
        
    return df.with_columns(
        pl.col(col_names).fill_null(fill_expr))

def StandardScaler(df: pl.DataFrame, col_names: list[str]) -> pl.DataFrame:
    """Applies standard scaling (z-score)."""
    return df.with_columns(
        (pl.col(col_names) - pl.col(col_names).mean()) / pl.col(col_names).std()
    )

def MinMaxScaler(df: pl.DataFrame, col_names: list[str]) -> pl.DataFrame:
    """Applies min-max scaling to range [0, 1]."""
    return df.with_columns(
        # This was the corrected line:
        (pl.col(col_names) - pl.col(col_names).min()) / (pl.col(col_names).max() - pl.col(col_names).min())
    )

def RobustScaler(df: pl.DataFrame, col_names: list[str]) -> pl.DataFrame:
    """Applies robust scaling using interquartile range."""
    return df.with_columns(
        # This was the corrected line:
        (pl.col(col_names) - pl.col(col_names).median()) / (pl.col(col_names).quantile(0.75) - pl.col(col_names).quantile(0.25))
    )

def LabelEncoder(df: pl.DataFrame, col_names: list[str]) -> pl.DataFrame:
    """
    Encodes categorical columns into integers.
    This is identical to an OrdinalEncoder in Polars.
    """
    return df.with_columns(
        pl.col(col_names).cast(pl.Categorical).to_physical()
    )


TRANSFORMER_MAP = {
    "SimpleImputer": SimpleImputer,
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
    "RobustScaler": RobustScaler,
    "LabelEncoder": LabelEncoder
}


def transformation_pipeline(analysis, df):
    """Create a sklearn Pipeline from an Analysis object."""
    df_transformed = df
    
    print("No. of Null values present: ", df_transformed.null_count())
    # Add transformations
    if analysis.transformations:
        for transform in analysis.transformations:
            transformer_func = TRANSFORMER_MAP[transform.transformer]
            columns_to_transform = transform.columns
            function_params = transform.parameters
            try:
                df_transformed = transformer_func(
                    df_transformed, 
                    col_names=columns_to_transform, 
                    **function_params)
                print(f"\nApplied {transform.transformer} to {columns_to_transform}")
            except Exception as e:
                print(f"Error applying {transform.transformer} to {columns_to_transform}: {e}")                   
    SimpleImputer(df_transformed, df_transformed.columns, strategy="bfill")
    print("No. of Null values present: ", sum(df_transformed.null_count()))
    print("Data Transformation done...")
    return df_transformed