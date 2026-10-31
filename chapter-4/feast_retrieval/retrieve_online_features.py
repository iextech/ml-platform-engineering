from datetime import datetime
import pandas as pd
from feast import FeatureStore

from feastentity import user

import os
from dotenv import load_dotenv

### Load environment variables
load_dotenv()

store = FeatureStore(repo_path=".")

inference_df = store.get_online_features(
    entity_rows=[{"user_id": "74b0638a-5742-4d1a-a03f-483794d7aa7c"}],
    features=[
        "demographic:Sex",
    ],
).to_df()

print("----- Feature schema -----\n")
print(inference_df.info())

print("----- Example features -----\n")
print(inference_df.head())
