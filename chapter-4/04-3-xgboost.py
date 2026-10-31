
import pandas as pd

import os 
from dotenv import load_dotenv

from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

from minio import Minio

import mlflow
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb

from util_functions import save_df_to_s3

### Load env variables
load_dotenv()

### Load input data & process
df=pd.read_csv('./data/income_data.csv', index_col=False)

target=df.Target
feature_df=df.drop('Target', axis=1)

encoder = OneHotEncoder(sparse_output=False, drop="if_binary")
target=encoder.fit_transform(np.array(target).reshape(-1,1))    

dummyfied_df = pd.get_dummies(feature_df, drop_first=True, sparse=False, dtype=float)
col_list = dummyfied_df.columns.to_list()

with open('column_list.pkl', 'wb') as f:
     pickle.dump(col_list, f)

X_train, X_test, y_train, y_test = train_test_split(dummyfied_df, target, train_size=0.80, shuffle=True)

### mlflow
BUCKET_NAME = "mlflow-datasets"

rustfs_client = Minio(os.getenv("S3_ENDPOINT_URL"), access_key=os.getenv("AWS_ACCESS_KEY_ID"), secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"), secure=False)
found = rustfs_client.bucket_exists(BUCKET_NAME)

mlflow.set_tracking_uri(os.getenv("TRACKING_URI"))
mlflow.set_experiment("income-classifier-new")

mlflow.xgboost.autolog(log_models=False)

configs = [
    {"objective":"binary:logistic", "num_boost_round": 30},
    {"objective":"binary:hinge", "num_boost_round": 10},
]

with mlflow.start_run(run_name="xgb-comparison"):
    for cfg in configs:
        params = {"objective": cfg["objective"], 'colsample_bytree': 1,'learning_rate': 1, 'max_depth': 10 , 'subsample':1}
        with mlflow.start_run(run_name=cfg["objective"], nested=True):
            n_round = cfg["num_boost_round"]
            dtrain  = xgb.DMatrix(data=X_train, label=y_train.ravel())
            dtest   = xgb.DMatrix(data=X_test, label=y_test.ravel())
            
            model   = xgb.train(params, dtrain, n_round)
            ax      = xgb.plot_importance(model, max_num_features=10, importance_type='cover')
            fig     = ax.figure
            fig.set_size_inches(10, 8)

            #pred_train = model.predict(dtrain)
            #pred_test  = model.predict(dtest)

            pred_train = (model.predict(dtrain) >= 0.5).astype(int)
            pred_test  = (model.predict(dtest)  >= 0.5).astype(int)

            roc_auc_score_train = roc_auc_score(y_train==1, pred_train)
            roc_auc_score_test = roc_auc_score(y_test==1, pred_test)
            training_accuracy = accuracy_score(y_train, pred_train)
            test_accuracy = accuracy_score(y_test, pred_test)

            mlflow.log_metric("roc_auc_score_train",roc_auc_score_train)    #B
            mlflow.log_artifact("column_list.pkl")
 
            objective = cfg["objective"]
            model_name = "income-classifier-xgboost-" + objective.split(":")[-1]
            mlflow.xgboost.log_model(model, name=model_name)

            print(f'Roc Auc Score train: {roc_auc_score_train}  \n')
            mlflow.log_metric("roc_auc_score_test",roc_auc_score_test)
            print(f'Roc Auc Score test: {roc_auc_score_test}  \n')
            mlflow.log_metric("training_accuracy",training_accuracy)
            print(f'Accuracy train : {training_accuracy}')
            mlflow.log_metric("test_accuracy",test_accuracy)

