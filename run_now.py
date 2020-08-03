import mlflow

# MLflow project will run on Databricks
mlflow.set_tracking_uri("databricks")

# Run MLflow project on databricks
# cluster spec is specified in `job_spec_aws.json`
# assume experiment, e.g., "first" has been created on Databricks workspace, e.g., "/Users/yue.li@salesforce.com/"
mlflow.projects.run(uri="https://github.com/judithliatsf/mlflow-example#mrpc",
                    version="mrpc",
                    backend="databricks",
                    backend_config="job_spec_aws.json", 
                    parameters={"layer_name": "bert-base-uncased"}, 
                    experiment_name="/Users/yue.li@salesforce.com/mrpc")
