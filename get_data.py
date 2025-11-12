import yaml
import requests
from pprint import pprint
from sklearn.model_selection import train_test_split
import polars as pl

class DataLoader:
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
        
        target_col = schema['target_column']        # "congestion_surcharge", "airport_fee"
        drop_cols = schema.get('drop_columns', [])  # "tpep_pickup_datetime", "tpep_dropoff_datetime"
        X = df.drop(drop_cols + target_col)
        y = df[target_col]
        return X, y, schema['test_split']

    def get_data_info(df):
        print("Columns: ", df.columns)
        print("\nColumn dtype: ", df.schema.values())
        pprint(df.describe())
        
    def split_data(x, y, test_size: float):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
        print("Training data shape: ", X_train.shape)
        print("Testing data shape: ", X_test.shape)
        return X_train, X_test, y_train, y_test
    
# ------------------------------------------------------------------- #
if __name__ == "__main__":
    # Instantiate the DataLoader class
    data_loader = DataLoader
    x, y, test_split = data_loader.load_yaml_file(filepath="data.yaml")
    
    # data_loader.get_data_info(x)
    
    x_train, x_test, y_train, y_test = data_loader.split_data(x, y, test_split)
