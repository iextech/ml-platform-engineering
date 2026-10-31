
import pandas as pd

import os 
from dotenv import load_dotenv

from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

from minio import Minio

import mlflow
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

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

print(found)

mlflow.set_tracking_uri(os.getenv("TRACKING_URI"))
mlflow.set_experiment("income-classifier-new")

with mlflow.start_run() as run:
    results = pd.DataFrame( index=['Roc Auc Score test', 'Accuracy score train', 'Accuracy Score test', 'time to fit' ])
    tree = DecisionTreeClassifier()
    run_id = run.info.run_id

    feature_df_path = f"income-classifier-datasets/feature_df-{run_id}.csv"
    save_df_to_s3(rustfs_client, feature_df, BUCKET_NAME, feature_df_path)

    train_df = pd.concat([X_train, pd.Series(y_train.ravel())], axis=1)
    train_df_path = f"income-classifier-datasets/train-{run_id}.csv"
    save_df_to_s3(rustfs_client, train_df, BUCKET_NAME, train_df_path)

    test_df = pd.concat([X_test, pd.Series(y_test.ravel())], axis=1)
    test_df_path = f"income-classifier-datasets/test-{run_id}.csv"
    save_df_to_s3(rustfs_client, test_df, BUCKET_NAME, test_df_path)

    training_dataset = mlflow.data.from_pandas(train_df, source=f"s3://{BUCKET_NAME}/{train_df_path}")    
    test_dataset     = mlflow.data.from_pandas(test_df, source=f"s3://{BUCKET_NAME}/{test_df_path}")
    feature_dataset  = mlflow.data.from_pandas(feature_df, source=f"s3://{BUCKET_NAME}/{feature_df_path}")

    mlflow.log_input(training_dataset, context="training")
    mlflow.log_input(test_dataset, context="testing")
    mlflow.log_input(feature_dataset, context="reference")

    tree.fit(X_train, y_train.ravel())
    roc_auc_score_train = roc_auc_score(y_train == 1, tree.predict_proba(X_train)[:, 1])    
    roc_auc_score_test  = roc_auc_score(y_test == 1, tree.predict_proba(X_test)[:, 1])

    training_accuracy = tree.score(X_train, y_train)
    test_accuracy     = tree.score(X_test, y_test)

    mlflow.log_metric("roc_auc_score_train", roc_auc_score_train)
    print(f"Roc Auc Score train: {roc_auc_score_train}\n")

    mlflow.log_metric("roc_auc_score_test", roc_auc_score_test)
    print(f"Roc Auc Score test: {roc_auc_score_test}\n")

    mlflow.log_metric("training_accuracy", training_accuracy)
    print(f"Accuracy train: {training_accuracy}")

    mlflow.log_metric("test_accuracy", test_accuracy)
    print(f"Accuracy test: {test_accuracy}")

    mlflow.sklearn.log_model(tree, name="income-classifier-decisiontree", skops_trusted_types=["sklearn.tree._tree.Tree"])
    mlflow.log_params(tree.get_params())
