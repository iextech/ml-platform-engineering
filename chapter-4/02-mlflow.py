"""
"""
import mlflow
import uuid

import os
from dotenv import load_dotenv

import pandas as pd
import os
from plotnine import ggplot, geom_bar, aes, theme_bw, coord_flip, labs

### Load environment variables
load_dotenv()

### Load input data
df=pd.read_csv('./data/income_data.csv', index_col=False)


### MLflow Call

mlflow.set_tracking_uri(os.getenv("TRACKING_URI"))
mlflow.set_experiment("income-classifier-new")

with mlflow.start_run(run_name=f"eda-{uuid.uuid4()}"):
    for i in df.iloc[:,:-1].select_dtypes(include='object').columns:
        print(f'Variable {i}  \n ')
        print(df[i].value_counts())
        plot = (
            ggplot(df)
            + geom_bar(
                aes(x=df[i], fill=df.Target),
                position='fill'
            )
            + theme_bw()
            + labs(title=f'Variable {i} ~ Target')
            + coord_flip()
        )
        print(plot)
        if not os.path.exists("categorical_variable_plots"):
            os.makedirs("categorical_variable_plots")
        plot.save(f"categorical_variable_plots/Variable {i}")
        mlflow.log_artifacts("categorical_variable_plots")
