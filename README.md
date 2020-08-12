# Example MLflow project

Examples of how to use [MLflow Projects](https://www.mlflow.org/docs/latest/projects.html) to manage ML lifecycles and stay away from unpredictable notebooks

### Setup Databricks CLI

1. Install MLflow: `pip install mlflow`
2. Install and configurate [Databricks CLI](https://docs.databricks.com/dev-tools/cli/index.html): `pip install databricks-cli`

### Run MLflow Project
1. Create a MLflow Experiment on Databricks UI
2. Git clone the current repo
3. Change the MLflow Experiment name in `run_now.py`
4. `python run_now.py`
