## mlops_pipeline
Automated machine learning pipeline that handles the entire ML lifecycle.


This project uses a Generative AI (Google's Gemini) to automatically analyze a dataset's schema and generate a robust, executable `sklearn` preprocessing and modeling pipeline.

It bridges the gap between a high-level data science goal (e.g., "I want to perform regression") and a production-ready `sklearn.pipeline.Pipeline` object, using Pydantic for strict data validation and error handling.

## 💡 How It Works: The Planner-Validator-Executor Model

1.  **The Planner :**
    * We send the dataset's schema (columns, data types) and our goal (e.g., "regression") to the Gemini model.
    * We use **Pydantic's** `model_json_schema()` to force the AI to return its plan in a *strict, pre-defined JSON format* (the `Analysis` model).
    * The plan includes a data overview, a list of transformations, and model recommendations for the task.

2.  **The Validator (Python):**
    * The JSON plan is loaded back into our Pydantic `Analysis` model, which automatically validates all data types.
    * A custom validation function (`validate_and_fix_plan`) runs to correct common mistakes and `sklearn` conflicts:
        * Replaces `LabelEncoder` with `OrdinalEncoder` (since `LabelEncoder` breaks `ColumnTransformer`).
        * Sets `sparse_output=False` for `OneHotEncoder` to prevent errors with pandas output.
        * Removes any transformations for columns that aren't in the feature set (e.g., the target variable).
    * Will find another solution for this later on.

3.  **The Executor (Python):**
    * The validated plan is passed to the `data_pipeline.py` module.
    * It maps the string names from the plan (e.g., `"SimpleImputer"`) to the actual `sklearn` classes (e.g., `SimpleImputer()`).
    * It dynamically builds a `ColumnTransformer` and assembles the final, runnable `sklearn.pipeline.Pipeline` object.

This allows Gemini to do the creative "planning" while our Python code provides the "scaffolding" and "guardrails" to guarantee a working result.

## ✨ Key Features

* **Generated Pipelines:** Automatically suggests and builds preprocessing steps (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`, etc.) tailored to your data's schema.
* **Pydantic Validation:** Uses `pydantic` to enforce a strict JSON schema for the output, eliminating flaky or malformed responses.
* **Robust Error Handling:** Includes a validation layer to programmatically fix known conflicts between Gemini's suggestions and `sklearn` requirements.
* **Dynamic & Extensible:** Easily add new transformers or models by updating the Pydantic `Literal` types in `data_analysis.py` and adding them to the `TRANSFORMER_MAP` in `data_pipeline.py`.
* **Separation of Concerns:**
    * `data_analysis.py`: Defines the data structure and communication with the AI.
    * `data_pipeline.py`: Handles the `sklearn` logic and pipeline construction.
    * `main.py`: Orchestrates the entire flow from data loading to training.


## ⚙️ Installation

1.  **Clone the repository and open the repo in your code editor:**
    ```bash
    git clone https://github.com/AmishKakka/mlops_pipeline.git
    cd mlops_pipeline
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API Key:**
    Create a `.env` file in the root directory and add your API key:
    ```
    API_Key="your_google_api_key_here"
    ```
    Or in your terminal do this:
    ```
    export API_Key="your_google_api_key_here"
    ```

## 🚀 Usage

1.  **Configure your data:** Update `data.yaml` or modify the `load_yaml_file` function in `data_ingestion.py` to point to your dataset.
2.  **Run the analysis:**
    ```bash
    python main.py
    ```

The script will:
1.  Load and clean the data.
2.  Send the schema to Gemini to get an `Analysis` plan.
3.  Validate and fix the plan.
4.  Build a transformation pipeline and apply it.
5.  Build a model pipeline and train it on the transformed data.
6.  Print the results.