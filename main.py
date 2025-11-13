from data_analysis import Analysis, get_analysis
from data_ingestion import load_yaml_file, get_data_info

# ------------------------------------------------------------------- #
if __name__ == "__main__":
    # Load data and schema
    df, schema = load_yaml_file(filepath="data.yaml")
    
    # Display data information
    df_schema = get_data_info(df)
    
    # Analyze the dataset using Gemini 2.5 Flash-lite 
    get_analysis(df_schema)