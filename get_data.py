import yaml
import requests
import polars as pl

def load_yaml_file(filepath: str):
    config = yaml.safe_load(open(filepath))
    data_config = config['data']
    schema = data_config['schema']
    
    if data_config['source_type'] == "csv":
        df = pl.read_csv(data_config['path'], **data_config.get('read_params', {}))
    elif data_config['source_type'] == "parquet":
        df = pl.read_parquet(data_config['path'])
    elif data_config['source_type'] == "excel":
        df = pl.read_excel(data_config['path'], **data_config.get('read_params', {}))

    target_col = schema['target_column']
    drop_cols = schema.get('drop_columns', [])
    X = df.drop(drop_cols + target_col)
    y = df[target_col]
    return X, y, schema['test_split']
    
if __name__ == "__main__":
    load_yaml_file("data.yaml")