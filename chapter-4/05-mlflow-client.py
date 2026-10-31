import os
import mlflow
from dotenv import load_dotenv

import mlflow
from mlflow import MlflowClient

### Load env variables
load_dotenv()

mlflow.set_tracking_uri(os.getenv("TRACKING_URI"))

mlflow_client = MlflowClient(tracking_uri=os.getenv("TRACKING_URI"))

experiment_name = "income-classifier-new"
experiment = mlflow_client.get_experiment_by_name(experiment_name)

run_object = mlflow_client.search_runs(experiment_ids=experiment.experiment_id,filter_string="metrics.roc_auc_score_test > 0.8",max_results=1,order_by=["metrics.roc_auc_score_test DESC"])[0]

run_id = f"{run_object.info.run_id}"
models = mlflow.search_logged_models(filter_string=f"source_run_id = '{run_id}'", experiment_ids=[mlflow.get_run(run_id).info.experiment_id], output_format="list")

if len(models) != 1:
    print("[ERROR] : The expected number of models is 1. The number is " + len(models)) 
    exit

model_uri = "models:/" + (models[0].model_id)
print(model_uri)

mlflow.register_model(model_uri, "random-forest-classifier")

