import mlflow

# MLflow project will run on Databricks
mlflow.set_tracking_uri("databricks")

# Run MLflow project on databricks
# cluster spec is specified in `job_spec_aws.json`
# assume experiment, e.g., "first" has been created on Databricks workspace, e.g., "/Users/yue.li@salesforce.com/"

# Option 1: MLflow Project root from Github uri (include github integration)
# mlflow.projects.run(uri="https://github.com/judithliatsf/mlflow-example#mrpc",
#                     version="mrpc",
#                     backend="databricks",
#                     backend_config="mrpc/job_spec_aws.json", 
#                     parameters={"epochs": 3, "train_steps": 200}, 
#                     experiment_name="/Users/yue.li@salesforce.com/mrpc")

# Option 2: MLflow Project root from local directory
mlflow.projects.run(uri="./",
                    backend="databricks",
                    backend_config="training/job_spec_aws.json", 
                    parameters={"epochs": 1, "train_steps": 10}, 
                    experiment_name="/Users/yue.li@salesforce.com/mrpc")

# Option 3: Run MLflow locally
# mlflow.projects.run(uri="./",
#                     parameters={"epochs": 1, "train_steps": 1},
#                     use_conda=False,
#                     experiment_name="mrpc")
