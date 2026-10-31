# Overview
This is an updated version of https://github.com/practical-mlops/chapter-4 project for the sub01 cluster. The description of the sub01 cluster is available at https://ml-engineering.cis-exp.cluster-wizard.com/sub01.html.

# Environment Setup
```bash
uv venv
source .venv/bin/activate
deactivate
```

## Package Install Commands 
- mlflow==3.17.2
- plotnine
- boto3
- minio
- pip
- xgboost
- feast
- feast[redis]
- s3fs

The install command is `uv pip install [package]`. The full list of the installed package is at the end of this documentation.

## Bucket and Object Preparation
- Buckets: mlflow-datasets, feature-data-sets, feature-registry
- Objects: Copy the three files in `feast/feature_data` to `feature-data-sets` bucket

## How to run python programs.
Please note that these files are not well polished. 

### ML 
```bash
uv run python [python file]
```

- 01-data-exploration.py
- 02-mlflow.py 
- 03-data-preparation.py
- 04-2-randomforest.py
- 04-3-xgboost.py
- 05-mlflow-client.py
### Feast
```bash
feast apply
START_TIME="2022-09-16T00:00:00"; END_TIME="2023-09-17T00:00:00"
uv run --env-file .env feast materialize $START_TIME $END_TIME
```
This command need to be executed in the `feast` directory. 
The second command need to be executed before running `feast_retrieval/retrieve_online_features.py`.

```bash
uv run python retrieve_features.py
uv run python retrieve_online_features.py
```


## Installed Python Packages

| Package | Version |
|---|---|
| aiobotocore | 3.10.0 |
| aiohappyeyeballs | 2.7.1 |
| aiohttp | 3.14.5 |
| aioitertools | 0.13.0 |
| aiosignal | 1.4.0 |
| alembic | 1.20.1 |
| annotated-doc | 0.0.5 |
| annotated-types | 0.8.0 |
| anyio | 4.15.1 |
| argon2-cffi | 25.1.0 |
| argon2-cffi-bindings | 26.1.0 |
| ast-serialize | 0.13.1 |
| async-timeout | 5.0.1 |
| attrs | 26.1.0 |
| bigtree | 1.5.3 |
| blinker | 1.9.0 |
| boto3 | 1.43.123 |
| botocore | 1.43.117 |
| cachetools | 7.2.7 |
| certifi | 2026.10.23 |
| cffi | 2.1.1 |
| charset-normalizer | 3.5.2 |
| click | 8.5.0 |
| cloudpickle | 3.1.2 |
| colorama | 0.4.6 |
| contourpy | 1.3.2 |
| cryptography | 50.0.2 |
| cycler | 0.12.1 |
| dask | 2026.8.0 |
| databricks-sdk | 0.158.0 |
| dill | 0.4.1 |
| docker | 7.2.0 |
| exceptiongroup | 1.3.1 |
| fastapi | 0.162.0 |
| feast | 0.67.0 |
| flask | 3.1.3 |
| flask-cors | 6.0.5 |
| fonttools | 4.65.0 |
| formulaic | 1.2.2 |
| frozenlist | 1.8.0 |
| fsspec | 2026.9.0 |
| gitdb | 4.0.12 |
| gitpython | 3.2.1 |
| google-auth | 2.62.0 |
| graphene | 3.4.3 |
| graphql-core | 3.2.13 |
| graphql-relay | 3.2.0 |
| greenlet | 3.5.6 |
| gunicorn | 26.2.0 |
| h11 | 0.16.0 |
| hiredis | 3.5.0 |
| httptools | 0.9.0 |
| huey | 3.5.3 |
| idna | 3.20 |
| importlib-metadata | 9.0.1 |
| interface-meta | 2.0.1 |
| itsdangerous | 2.2.0 |
| jinja2 | 3.1.6 |
| jmespath | 1.1.0 |
| joblib | 1.6.0 |
| kiwisolver | 1.5.1 |
| librt | 0.16.0 |
| locket | 1.0.0 |
| mako | 1.4.3 |
| markupsafe | 3.0.4 |
| matplotlib | 3.10.9 |
| minio | 7.2.20 |
| mizani | 0.14.6 |
| mlflow | 3.17.2 |
| mlflow-skinny | 3.17.2 |
| mlflow-tracing | 3.17.2 |
| mmh3 | 5.3.1 |
| multidict | 7.1.0 |
| mypy | 2.4.1 |
| mypy-extensions | 1.1.0 |
| narwhals | 2.28.0 |
| numpy | 2.2.6 |
| nvidia-nccl-cu12 | 2.32.3 |
| opentelemetry-api | 1.45.2 |
| opentelemetry-proto | 1.45.2 |
| opentelemetry-sdk | 1.45.2 |
| opentelemetry-semantic-conventions | 0.66b2 |
| packaging | 26.3 |
| pandas | 2.3.3 |
| partd | 1.4.2 |
| pathspec | 1.1.1 |
| patsy | 1.0.3 |
| pillow | 12.3.0 |
| pip | 26.2.2 |
| plotnine | 0.15.8 |
| prettytable | 3.18.0 |
| prometheus-client | 0.24.1 |
| propcache | 0.5.4 |
| protobuf | 7.36.2 |
| psutil | 7.2.2 |
| pyarrow | 25.0.1 |
| pyasn1 | 0.6.5 |
| pyasn1-modules | 0.4.2 |
| pycparser | 3.11 |
| pycryptodome | 4.2.0 |
| pydantic | 2.14.2 |
| pydantic-core | 2.50.2 |
| pygments | 2.21.0 |
| pyjwt | 2.15.1 |
| pyparsing | 3.3.3 |
| python-dateutil | 2.9.0.post0 |
| python-dotenv | 1.2.4 |
| pytz | 2026.6 |
| pyyaml | 6.0.3 |
| redis | 7.4.1 |
| requests | 2.34.2 |
| s3fs | 2026.9.0 |
| s3transfer | 0.19.3 |
| scikit-learn | 1.7.2 |
| scipy | 1.15.3 |
| six | 1.17.0 |
| skops | 0.16.0 |
| smmap | 5.0.3 |
| sqlalchemy | 2.0.55 |
| sqlparse | 0.6.0 |
| starlette | 1.7.0 |
| statsmodels | 0.15.0 |
| tabulate | 0.10.0 |
| tenacity | 8.5.0 |
| threadpoolctl | 3.7.0 |
| toml | 0.10.2 |
| tomli | 2.5.0 |
| toolz | 1.2.0 |
| tqdm | 4.70.1 |
| typeguard | 4.6.0 |
| typing-extensions | 4.16.0 |
| typing-inspection | 0.4.4 |
| tzdata | 2026.6 |
| urllib3 | 2.8.0 |
| uvicorn | 0.54.0 |
| uvicorn-worker | 0.4.0 |
| uvloop | 0.23.0 |
| watchfiles | 1.3.0 |
| wcwidth | 0.9.4 |
| websockets | 16.1.1 |
| werkzeug | 3.1.9 |
| wrapt | 2.5.2 |
| xgboost | 3.2.0 |
| yarl | 1.25.1 |
| zipp | 4.1.2 |

