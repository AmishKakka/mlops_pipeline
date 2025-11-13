import yaml
import requests
from pprint import pprint
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
    elif data_config['source_type'] == "api":
        if 'api' not in data_config:
            raise ValueError("API endpoint not specified in data configuration.")
        else:
            response = requests.get(data_config['api'])
            response.raise_for_status()
            print("Data fetched successfully from API.")
            with open("temp_data.parquet", "wb") as f:
                f.write(response.content)
            df = pl.read_parquet("temp_data.parquet")
    else:
        raise ValueError(f"Unsupported source type: {data_config['source_type']}")
    return df, schema

def get_data_info(df):
    # print("Columns: ", df.columns)
    # print("\nColumn dtype: ", df.schema.values())
    # pprint(df.describe())
    return df.schema
    
    
# ------------------------------------------------------------------- #
if __name__ == "__main__":
    df, schema = load_yaml_file(filepath="data.yaml")
    
    # data_loader.get_data_info(x)