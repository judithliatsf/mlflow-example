import mlflow

# MLflow project will run on Databricks
mlflow.set_tracking_uri("databricks")

# Run MLflow project on databricks
# cluster spec is specified in `job_spec_aws.json`
# assume experiment, e.g., "first" has been created on Databricks workspace, e.g., "/Users/yue.li@salesforce.com/"

# Option 1: MLflow Project root from Github uri (include github integration)
# mlflow.projects.run(uri="https://github.com/judithliatsf/mlflow-example",
#                     version="mrpc",
#                     backend="databricks",
#                     backend_config="training/job_spec_aws.json",
#                     parameters={"epochs": 2, "train_steps": 117},
#                     experiment_name="/Users/yue.li@salesforce.com/mrpc")

# Option 2: MLflow Project root from local directory
mlflow.projects.run(uri="training/",
                    backend="databricks",
                    backend_config="training/job_spec_aws.json",
                    parameters={"epochs": 1, "train_steps": 5},
                    experiment_name="/Users/yue.li@salesforce.com/mrpc")

# Option 3: Run MLflow locally
# mlflow.projects.run(uri="training/",
#                     parameters={"epochs": 1, "train_steps": 1},
#                     use_conda=False,
#                     experiment_name="mrpc")
