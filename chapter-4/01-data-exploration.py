"""
   Data exploration : 
"""

import pandas as pd
import os
from plotnine import ggplot, geom_bar, aes, theme_bw, coord_flip, labs

### Data investigation
df=pd.read_csv('./data/income_data.csv', index_col=False)
print("income data head", '\n')
print(df.head(), '\n')
print("income data info", '\n')
print(df.info(), '\n')
print("income data describe", '\n')
print(df.describe(), '\n')

### Plots for data
if not os.path.exists("categorical_variable_plots"):
    os.makedirs("categorical_variable_plots")    

for i in df.iloc[:,:-1].select_dtypes(include='object').columns:
    print(f'Variable {i}  \n ')
    print(df[i].value_counts())
    plot = ggplot(df)+ geom_bar(aes(x=df[i], fill=df.Target),  position='fill') + theme_bw() + labs(title=f'Variable {i} ~ Target') + coord_flip()
    print(plot)

    plot.save(f"categorical_variable_plots/Variable {i}")
