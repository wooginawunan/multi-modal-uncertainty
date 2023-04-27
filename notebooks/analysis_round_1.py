# %% Configuration
from dotenv import load_dotenv
load_dotenv('../env.sh')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np


# Environment variables
PATH = os.environ['RESULTS_DIR']
experiments = ['Vanilla/0.1', 'Vanilla/0.1_0', 'Vanilla/0.1_1',
               'single_model_weight_sharing/0.1', 'single_model_weight_sharing/0.1_0', 'single_model_weight_sharing/0.1_1',
               'MultiHead/0.1', 'MultiHead/0.1_0', 'MultiHead/0.1_1',
               'MIMO_shuffle_instance/0.1', 'MIMO_shuffle_instance/0.1_0', 'MIMO_shuffle_instance/0.1_1',
               'MIMO_shuffle_view/0.1', 'MIMO_shuffle_view/0.1_0', 'MIMO_shuffle_view/0.1_1',
               'MIMO_shuffle_all/0.1', 'MIMO_shuffle_all/0.1_0', 'MIMO_shuffle_all/0.1_1', 
               ]
lr = 0.1


### OVERVIEW: performance and learning curves for the first round of experiments
# Read in the data

all_dfs = []
for exp in experiments:
    df = pd.read_csv(os.path.join(PATH, exp, 'history.csv'))
    df['model_type'], df['seed'] = exp.split('/')
    all_dfs.append(df)
    
all_dfs = pd.concat(all_dfs)


# %% Loading history files
sns.set_theme(style="whitegrid")
sns.set_context("paper", 
                font_scale=1.5, 
                rc={"lines.linewidth": 2.5})

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
sns.lineplot(x="epoch", y="loss",
             hue="model_type", style="seed", 
             ax = axs[0, 0],
             data=all_dfs)

sns.lineplot(x="epoch", y="test_loss",
             hue="model_type", style="seed", 
             ax = axs[0, 1],
             data=all_dfs)

sns.lineplot(x="epoch", y="acc",
             hue="model_type", style="seed", 
             ax = axs[1, 0],
             data=all_dfs)

sns.lineplot(x="epoch", y="test_acc",
             hue="model_type", style="seed", 
             ax = axs[1, 1],
             data=all_dfs)
plt.tight_layout()
plt.savefig('round_1_learning_curves.png')
# %%

all_dfs.groupby(['model_type', 'seed'])[['test_acc']].max().reset_index().groupby('model_type').agg(['mean', 'std'])

### Prediction Diversity Analysis 

# %%
import scipy.stats as stats
import itertools 

def trunk_pred_top(pred, test_cls, top, mute_true=False):
    pred_ = []
    for i in range(len(pred)):
        p = pred[i].copy()
        if mute_true:
            p[test_cls[i]] = 0
        
        value = np.partition(pred[i].flatten(), -top)[-top]
        p = [j if j>=value else 0 for j in p ]
        pred_.append(p)
    return np.array(pred_)

def subnetwork_wise_kendalltau(preds_muted):
    outputs = np.array([stats.kendalltau(x, y) \
        for x, y in itertools.combinations(preds_muted, 2)])
    return outputs[:, 0] 


results = {'accuracy_viewwise': [], 
           'accuracy_overall': [], 
           'kendalltau': []}
for exp in experiments[3:]:
    predictions = np.load(os.path.join(PATH, exp, 'model_best_val_predictions.npy'))
    # labels = np.load(os.path.join(PATH, exp, 'model_best_val_labels.npy'))  
    
    acc_overall = np.equal(np.argmax(predictions.mean(1), 1), labels).mean()
    acc_subnetworks = [(np.argmax(predictions[:, i, :], 1)==labels).mean()\
        for i in range(predictions.shape[1])]

    print(f'Experiment: {exp}')
    print(f'\tAccuracy overall: {acc_overall}')
    print(f'\tAccuracy per subnetwork: {acc_subnetworks}')

    top=5
    num_views = predictions.shape[1]
    preds_muted = [trunk_pred_top(predictions[:, i, :], labels, top, mute_true=True)\
        for i in range(num_views)]
    taus = subnetwork_wise_kendalltau(preds_muted)
    
    print(f'\tKendall Tau @ Top5 among subnetworks: {taus.mean()}')


# %%

# results = pd.DataFrame(results, index=experiments)
# results
# def cases_breakdown(labels, predictions):
#     churn_cases = []
#     suprising_cases = []
#     unsolved_cases = []
        
#     viewwise_correctness = [np.argmax(predictions[:, i,:], 1) == labels \
#         for i in range(predictions.shape[1])]

#     overall_correctness = np.argmax(predictions.mean(1), 1) == labels
    
#     suprising_cases.append(np.sum(~acc_0 & ~acc_1 & acc))
#     unsolved_cases.append(np.sum(~acc_0 & ~acc_1 & ~acc))
#     churn_cases.append(np.sum((acc_0 | acc_1) & ~acc))
    
#     return churn_cases, suprising_cases, unsolved_cases

# churn_cases, suprising_cases, unsolved_cases  = cases_breakdown(
#     labels, predictions)


# %%

### Prediction robustness analysis 



# %%
for exp in experiments:
    predictions = np.load(os.path.join(
        PATH, exp, 'model_best_val_predictions_robustness.npy'))
    #labels = np.load(os.path.join(PATH, exp, 'model_best_val_labels.npy')) 

    num_views = predictions.shape[0]
    acc_per_view = [(np.argmax(predictions[i, :, :, :].mean(1), 1)==labels).mean()\
        for i in range(num_views)]

    print(f'Experiment: {exp}')
    
    for i in range(num_views):
        print(f'\tAccuract when missing view {i}: {acc_per_view[i]}')
        
        if 'Vanilla' in exp:
            continue
        else:
            top=5
            preds_muted = [trunk_pred_top(predictions[i, :, j, :], 
                                        labels, top, mute_true=True)\
                for j in range(num_views-1 if 'single_model_weight_sharing' in exp\
                    else num_views)]
            taus = subnetwork_wise_kendalltau(preds_muted)
            print(f'\t\tKendall Tau @ Top5 among subnetworks: {taus.mean()}')
            print('\t\tPairwise Kendall Tau @ Top5:')       
            if exp == 'single-model-weight-sharing':
                continue 
            for x, y in zip([p for p in itertools.combinations("1234", 2)], taus):
                print('\t\t', x, y) 
              
# %%
