# %% Analyze the robustness 
from dotenv import load_dotenv
load_dotenv('../env.sh')

import pandas as pd
import seaborn as sns
import os
import numpy as np
from sklearn.metrics import roc_auc_score
import seaborn as sns

# Environment variables
PATH = os.environ['RESULTS_DIR']

sns.set_theme(style="whitegrid")
from utils import (get_correlation, 
                    visuals_per_model,
                    load_robustness_experiment_results, 
                    plot_correlation_and_performance,
                    softmax)

def AUC_table(labels, ori, image, text, image_correspondence, text_correspondence):
    image_control = np.array([roc_auc_score(labels, image_correspondence[:, i]) 
                              for i in range(20)])
    text_control = np.array([roc_auc_score(labels, text_correspondence[:, i]) 
                             for i in range(20)])
    
    df =  pd.DataFrame({   
        'variants': ['full', 'image', 'text'],
        'AUC': [roc_auc_score(labels, ori), 
                roc_auc_score(labels, image), 
                roc_auc_score(labels, text)],
    })

    for i, auc in enumerate(image_control):
        df.loc[i + 3] = ['image_control', auc]
    for i, auc in enumerate(text_control):
        df.loc[i + 23] = ['text_control', auc]

    print(df.groupby('variants')['AUC'].agg(['mean', 'std']))
    return df


# %% Experiment with different hyperparameters
# checkpoint_name = 'model_best_val'
# phase = 'test'

# for exp in ['head3_layer3/clip_transformer/Vanilla/128_0.1',
#             'head3_layer3/clip_transformer/Vanilla/32_0.01',
#             'head3_layer3/clip_transformer/Vanilla/128_0.01',
#             ]:


#     predictions = np.load(os.path.join(
#         PATH, exp, f"robustness_{checkpoint_name}_predictions_{phase}.npy"))

#     labels = np.load(os.path.join(
#         PATH, exp, f"robustness_{checkpoint_name}_labels_{phase}.npy"))

#     ori = softmax(predictions[:, 0, :]).mean(1)[:, 1]
#     image = softmax(predictions[:, 1, :]).mean(1)[:, 1]
#     text = softmax(predictions[:, 2, :]).mean(1)[:, 1]
#     image_correspondence = softmax(predictions[:, 3:3+20, :]).mean(2)[:, :, 1]
#     text_correspondence = softmax(predictions[:, 3+20:, :]).mean(2)[:, :, 1]

#     aucs = AUC_table(labels, ori, image, text, image_correspondence, text_correspondence)
#     aucs['experiment'] = exp

#     #histogram_by_group(labels, ori, image, text, image_correspondence, text_correspondence)
#     sns.set_theme(style="whitegrid")
#     sns.set_context("talk")
#     fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=False)
#     violin_plot_by_group(axs[0], labels, ori, image, text, image_correspondence, text_correspondence)
#     scatter_plot_instance_level(axs[1], labels, ori, image, text, image_correspondence, text_correspondence)
#     fig.tight_layout()
#     fig.show()
#     fig.savefig(
#         os.path.join('hatefulmeme', f"robustness_{checkpoint_name}_violin_{phase}_{exp.replace('/', '_')}.pdf"))
    

# focus on the analysis on the vanilla model
    # if 'Vanilla' not in exp:
    #     for j in [0, 1]:
    #         ori = softmax(predictions[:, 0, :])[:, j, 1]
    #         image = softmax(predictions[:, 1, :])[:, j, 1]
    #         text = softmax(predictions[:, 2, :])[:, j, 1]
    #         image_correspondence = softmax(predictions[:, 3:3+20, :])[:, :, j,  1]
    #         text_correspondence = softmax(predictions[:, 3+20:, :])[:, :, j, 1]

    #         aucs = AUC_table(labels, ori, image, text, image_correspondence, text_correspondence)
    #         aucs['experiment'] = exp
    
    #         histogram_by_group(labels, ori, image, text, 
    #                                     image_correspondence, text_correspondence)
            
    #         violin_plot_by_group(labels, ori, image, text, 
    #                                     image_correspondence, text_correspondence)
            

    #         scatter_plot_instance_level(labels, ori, image, text, 
    #                                     image_correspondence, text_correspondence)

# %% Timewise analysis

def process_predictions_hatefulmeme(predictions, labels):
    ori = softmax(predictions[:, 0, :]).mean(1)[:, 1]
    image = softmax(predictions[:, 1, :]).mean(1)[:, 1]
    text = softmax(predictions[:, 2, :]).mean(1)[:, 1]
    image_correspondence = softmax(predictions[:, 3:3+20, :]).mean(2)[:, :, 1]
    text_correspondence = softmax(predictions[:, 3+20:, :]).mean(2)[:, :, 1]

    return labels, ori, image, text, image_correspondence, text_correspondence

def epoch_wise_analysis(phase, exp, epochs, make_plots=True):
    
    dataset = 'hateful-meme'

    results_auc = []
    results_corr = []

    for epoch in epochs:
        checkpoint_name = f'model_epoch_{epoch}'
        predictions, labels = load_robustness_experiment_results(
            checkpoint_name, phase, exp, dataset)
            
        outcomes = process_predictions_hatefulmeme(
            predictions, labels)

        aucs = AUC_table(*outcomes)

        aucs['epoch'] = epoch
        results_auc.append(aucs) 

        # evaluate by correlation per group: image, text
        corr = get_correlation(*outcomes) # as a dict
        corr['epoch'] = epoch
        results_corr.append(corr)

        # do the visualizations of the correlations
        if make_plots:
            save_folder = os.path.join(
                dataset, 'robustness_per_epoch', exp.replace('/', '_'), phase)
            if os.path.exists(save_folder) is False:
                os.makedirs(save_folder)
            
            visuals_per_model(outcomes, save_folder, checkpoint_name)
    
    # concatenate the results
    results_auc = pd.concat(results_auc, ignore_index=True)

    results_corr = pd.DataFrame(results_corr)
    results_corr.index = results_corr.epoch
    results_corr = results_corr.drop('epoch', axis=1)

    return results_auc, results_corr


# %%
phase = 'train'
exp = 'head3_layer3/clip_transformer/Vanilla/128_0.01_save_all_checkpoints'
epochs = range(1, 100)

results_auc, results_corr = epoch_wise_analysis(phase, exp, epochs, make_plots=False)

group_by_table = results_auc.groupby(['variants', 'epoch']).mean().reset_index()
full = group_by_table[group_by_table['variants']=='full'].AUC.values
text = group_by_table[group_by_table['variants']=='text'].AUC.values
image = group_by_table[group_by_table['variants']=='image'].AUC.values
image_control = group_by_table[group_by_table['variants']=='image_control'].AUC.values
text_control = group_by_table[group_by_table['variants']=='text_control'].AUC.values

plot_correlation_and_performance(results_corr, full, image, text, 'AUROC')

# %%
exp = 'head3_layer3/clip_transformer/Vanilla/128_0.1_save_all_checkpoints'
epochs = range(1, 100)

results_auc, results_corr = epoch_wise_analysis(phase, exp, epochs, make_plots=False)

group_by_table = results_auc.groupby(['variants', 'epoch']).mean().reset_index()
full = group_by_table[group_by_table['variants']=='full'].AUC.values
text = group_by_table[group_by_table['variants']=='text'].AUC.values
image = group_by_table[group_by_table['variants']=='image'].AUC.values
image_control = group_by_table[group_by_table['variants']=='image_control'].AUC.values
text_control = group_by_table[group_by_table['variants']=='text_control'].AUC.values

plot_correlation_and_auc(results_corr, full, image, text)


# %%
exp = 'head3_layer3/clip_transformer/Vanilla/128_0.1_save_all_checkpoints'
epochs = range(1, 100)

results_auc, results_corr = epoch_wise_analysis(phase, exp, epochs, make_plots=False)

group_by_table = results_auc.groupby(['variants', 'epoch']).mean().reset_index()
full = group_by_table[group_by_table['variants']=='full'].AUC.values
text = group_by_table[group_by_table['variants']=='text'].AUC.values
image = group_by_table[group_by_table['variants']=='image'].AUC.values
image_control = group_by_table[group_by_table['variants']=='image_control'].AUC.values
text_control = group_by_table[group_by_table['variants']=='text_control'].AUC.values

plot_correlation_and_auc(results_corr, full, image, text)

# %%
phase = 'test'
exp = 'head3_layer3/clip_transformer/Vanilla/32_0.001_save_all_checkpoints'
epochs = range(1, 57)

results_auc, results_corr = epoch_wise_analysis(phase, exp, epochs, make_plots=False)

sns.lineplot(data=results_auc, hue='variants', x='epoch', y='AUC')


# %%
phase = 'test'
exp = 'head3_layer3/clip_transformer/Vanilla/32_0.001_save_all_checkpoints'
epochs = range(1, 100)

results_auc, results_corr = epoch_wise_analysis(phase, exp, epochs, make_plots=False)

group_by_table = results_auc.groupby(['variants', 'epoch']).mean().reset_index()
full = group_by_table[group_by_table['variants']=='full'].AUC.values
text = group_by_table[group_by_table['variants']=='text'].AUC.values
image = group_by_table[group_by_table['variants']=='image'].AUC.values
image_control = group_by_table[group_by_table['variants']=='image_control'].AUC.values
text_control = group_by_table[group_by_table['variants']=='text_control'].AUC.values

plot_correlation_and_auc(results_corr, full, image, text)
sns.lineplot(data=results_auc, hue='variants', x='epoch', y='AUC')

# %%

def ensemble_overtime(epoches_to_ensemble, dataset='hateful_memes'):
    predictions = []
    for epoch in epoches_to_ensemble:
        checkpoint_name = f'model_epoch_{epoch}'
        labels, ori, image, text, image_correspondence, text_correspondence = \
            load_robustness_experiment_results(checkpoint_name, phase, exp, dataset)
        print('@Epoch', epoch, 'AUC=', roc_auc_score(labels, ori))
        predictions.append(ori)
    ensemble_predictions = np.array(predictions).mean(0)
    print(f'Ensemble of {epoches_to_ensemble}', 'AUC=', roc_auc_score(labels, ensemble_predictions))

ensemble_overtime(range(50, 60))
ensemble_overtime(range(80, 89))
ensemble_overtime(range(10, 20))
ensemble_overtime(range(20, 30))
# %%
group_by_table.iloc[group_by_table[group_by_table['variants']=='full'].AUC.idxmax()]
# %%
txt_epoch = group_by_table.iloc[group_by_table[group_by_table['variants']=='text'].AUC.idxmax()].epoch
img_epoch = group_by_table.iloc[group_by_table[group_by_table['variants']=='image'].AUC.idxmax()].epoch
ensemble_overtime([txt_epoch, img_epoch ])
# %%



phase = 'test'
exp = 'head6_layer6/clip_transformer/Vanilla/32_1e-5_save_all_checkpoints'
epochs = range(1, 30)

results_auc, results_corr = epoch_wise_analysis(phase, exp, epochs, make_plots=False)

group_by_table = results_auc.groupby(['variants', 'epoch']).mean().reset_index()
full = group_by_table[group_by_table['variants']=='full'].AUC.values
text = group_by_table[group_by_table['variants']=='text'].AUC.values
image = group_by_table[group_by_table['variants']=='image'].AUC.values
image_control = group_by_table[group_by_table['variants']=='image_control'].AUC.values
text_control = group_by_table[group_by_table['variants']=='text_control'].AUC.values

plot_correlation_and_auc(results_corr, full, image, text)
sns.lineplot(data=results_auc, hue='variants', x='epoch', y='AUC')
# %%


phase = 'test'
exp = 'head6_layer6/clip_transformer/Vanilla/32_0.001_save_all_checkpoints'
epochs = range(1, 100)

results_auc, results_corr = epoch_wise_analysis(phase, exp, epochs, make_plots=False)

group_by_table = results_auc.groupby(['variants', 'epoch']).mean().reset_index()
full = group_by_table[group_by_table['variants']=='full'].AUC.values
text = group_by_table[group_by_table['variants']=='text'].AUC.values
image = group_by_table[group_by_table['variants']=='image'].AUC.values
image_control = group_by_table[group_by_table['variants']=='image_control'].AUC.values
text_control = group_by_table[group_by_table['variants']=='text_control'].AUC.values

plot_correlation_and_auc(results_corr, full, image, text)
sns.lineplot(data=results_auc, hue='variants', x='epoch', y='AUC')

# %%



# %%
phase = 'test'
exp = 'head6_layer6/clip_transformer/MIMO-shuffle-instance/32_1e-5_save_all_checkpoints'
epochs = range(1, 9)

results_auc, results_corr = epoch_wise_analysis(phase, exp, epochs, make_plots=True)

group_by_table = results_auc.groupby(['variants', 'epoch']).mean().reset_index()
full = group_by_table[group_by_table['variants']=='full'].AUC.values
text = group_by_table[group_by_table['variants']=='text'].AUC.values
image = group_by_table[group_by_table['variants']=='image'].AUC.values
image_control = group_by_table[group_by_table['variants']=='image_control'].AUC.values
text_control = group_by_table[group_by_table['variants']=='text_control'].AUC.values

plot_correlation_and_auc(results_corr, full, image, text)
# %%
