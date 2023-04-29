# %%
from dotenv import load_dotenv
load_dotenv('../env.sh')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

PATH = os.environ['RESULTS_DIR'].replace('hateful-meme', '/food101')
experiments = [ '', #'MultiHead', 'MIMO-shuffle-instance'
               ]
prefix, suffix = 'head3_layer3/clip_transformer', '128_0.001'
### OVERVIEW: performance and learning curves for the first round of experiments
# Read in the data

all_dfs = []
for exp in experiments:
#    df = pd.read_csv(os.path.join(PATH, 'updated_encoder', exp, 'history.csv'))
    df = pd.read_csv(os.path.join(PATH, prefix, exp, suffix, 'history.csv'))
    df['model_type'] = exp
    all_dfs.append(df)
    
all_dfs = pd.concat(all_dfs)

sns.set_theme(style="whitegrid")
sns.set_context("paper", 
                font_scale=1.5, 
                rc={"lines.linewidth": 2.5})

fig, axs = plt.subplots(3, 3, figsize=(15, 8))
sns.lineplot(x="epoch", y="loss",
             hue="model_type",
             ax = axs[0, 0],
             data=all_dfs)

sns.lineplot(x="epoch", y="val_loss",
             hue="model_type",
             ax = axs[0, 1],
             data=all_dfs)

sns.lineplot(x="epoch", y="test_loss",
             hue="model_type",
             ax = axs[0, 2],
             data=all_dfs)

sns.lineplot(x="epoch", y="acc",
             hue="model_type",
             ax = axs[1, 0],
             data=all_dfs)

sns.lineplot(x="epoch", y="val_acc",
             hue="model_type",
             ax = axs[1, 1],
             data=all_dfs)

sns.lineplot(x="epoch", y="test_acc",
             hue="model_type",
             ax = axs[1, 2],
             data=all_dfs)


sns.lineplot(x="epoch", y="val_auc",
             hue="model_type",
             ax = axs[2, 1],
             data=all_dfs)

sns.lineplot(x="epoch", y="test_auc",
             hue="model_type",
             ax = axs[2, 2],
             data=all_dfs)

plt.tight_layout()
plt.savefig(f'hatefulmeme/learning_curves_{prefix.replace("/", "_")}_{suffix}.png')

all_dfs.groupby(['model_type'])[['val_acc', 'val_auc', 'test_acc', 'test_auc']].max().to_csv(f'hatefulmeme/performance_{prefix.replace("/", "_")}_{suffix}.csv')


# %%
