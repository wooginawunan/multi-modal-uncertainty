# %% Analyze the robustness 
from dotenv import load_dotenv
load_dotenv('../env.sh')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.metrics import roc_auc_score
import seaborn as sns
sns.set_theme(style="whitegrid")
from matplotlib.colors import to_rgb
from matplotlib.collections import PolyCollection

# Environment variables
PATH = os.environ['RESULTS_DIR']

def softmax(x):
    return(np.exp(x)/np.exp(x).sum(-1, keepdims=True))

def ACC_tabel(predictions, labels):
    ori = predictions[:, 0, :].argmax(-1)
    image = predictions[:, 1, :].argmax(-1)
    text = predictions[:, 2, :].argmax(-1)
    image_correspondence = predictions[:, 3:3+20, :].argmax(-1)
    text_correspondence = predictions[:, 3+20:, :].argmax(-1)

    print('accuracy with full inputs:', (ori==labels).mean())
    print('accuracy with image:', (image==labels).mean())
    print('accuracy (control group):', 
        (image_correspondence==np.expand_dims(labels, 1)).mean())
    print('accuracy with text:', (text==labels).mean())
    print('accuracy (control group):', 
        (text_correspondence==np.expand_dims(labels, 1)).mean())

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

from scipy.stats import pearsonr
def get_correlation(labels, ori, image, text, image_correspondence, text_correspondence):

    def correlation(exp, control):
        x = exp - ori
        y = (control - np.expand_dims(ori, 1)).mean(1)
        return pearsonr(x, y)[0]
    
    return {'image': correlation(image, image_correspondence),
            'text': correlation(text, text_correspondence)}

def scatter_plot_instance_level(ax, labels, ori, image, text, 
                                image_correspondence, text_correspondence):
    
    
    b = len(labels)
    x = image - ori
    y = (image_correspondence - np.expand_dims(ori, 1)).mean(1)
    std = (image_correspondence - np.expand_dims(ori, 1)).std(1)
    
    x_ = text - ori
    y_ = (text_correspondence - np.expand_dims(ori, 1)).mean(1)
    std_ = (text_correspondence - np.expand_dims(ori, 1)).std(1)

    data = pd.DataFrame({'experimental': np.concatenate((x, x_)),
                        'control': np.concatenate((y, y_)),
                        'std': np.concatenate((std, std_)),
                        'modal': np.concatenate((np.repeat('image', b), np.repeat('text', b)))
                        })

    sns.scatterplot(
        data=data,
        x="experimental", y="control",
        hue="modal", size="std",
        sizes=(10, 200),
        alpha=.5, palette="muted",
        ax = ax,
    )

    h,l = ax.get_legend_handles_labels()
    ax.legend(h[1:3],l[1:3], loc='upper left', 
              frameon=False)
    
    ax.set_ylim([-1, 1])
    ax.set_xlim([-1, 1])

    ax.plot([-1, 1], [-1 ,1], 'k--', color='black', alpha=0.5)
    
    ax.set_xlabel("experimental: $\Delta p$")
    ax.set_ylabel("control: $\Delta p$")

def histogram_by_group(labels, ori, image, text, 
                                image_correspondence, text_correspondence):
    
    sns.set_theme(style="whitegrid")
    sns.set_context("talk")
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].hist(image - ori, color='blue', alpha=0.5, label='image only', 
                density=True, bins=20)
    axs[0].hist((image_correspondence - np.expand_dims(ori, 1)).mean(1), 
            color='green', alpha=0.5,
            label='control group', density=True, bins=20)
    axs[0].set_xlabel('Change in prediction against model with full inputs')
    axs[0].set_ylabel('Probability density')
    axs[0].legend()


    axs[1].hist(text - ori, color='blue', alpha=0.5, label='text only', 
                density=True, bins=20)
    axs[1].hist((text_correspondence - np.expand_dims(ori, 1)).mean(1), 
            color='green', alpha=0.5,
            label='control group', density=True, bins=20)
    axs[1].set_xlabel('Change in prediction against model with full inputs')
    axs[1].set_ylabel('Probability density')
    axs[1].legend()
    
    plt.show()

def violin_plot_by_group(ax, labels, ori, image, text, 
                                image_correspondence, text_correspondence):
        # Draw a nested violinplot and split the violins for easier comparison

    b = len(labels)
    data = pd.DataFrame({
        'diff_p': np.concatenate(
            (image - ori, (image_correspondence - np.expand_dims(ori, 1)).mean(1),  
            text - ori, (text_correspondence - np.expand_dims(ori, 1)).mean(1)
            )),
        'modal': np.concatenate((np.repeat('image', b*2), np.repeat('text', b*2))),
        'group': np.concatenate((
            np.repeat('experimental', b), np.repeat('control', b), 
            np.repeat('experimental', b), np.repeat('control', b)))
        })

    sns.violinplot(data=data, 
                    y="modal",
                    x="diff_p", 
                    hue="group",
                    palette=['.3', '.9'], 
                    split=True, 
                    inner='quart', 
                    ax=ax,
                    linewidth=1,
                )
    h,l = ax.get_legend_handles_labels()
    ax.legend(h[0:3],l[0:3], loc='lower right', 
              frameon=False)
    
    # ax.set_xlim([-1, 1])
    # ax.vlines(0, ymin=-1, ymax=1, colors='black', alpha=0.5)
    colors = sns.color_palette('tab10')
    for ind, violin in enumerate(ax.findobj(PolyCollection)):
        rgb = to_rgb(colors[ind // 2])
        if ind % 2 != 0:
            rgb = 0.5 + 0.5 * np.array(rgb)  # make whiter
        violin.set_facecolor(rgb)

    ax.set_xlabel("$\Delta p$")
    ax.set_ylabel("")

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

def load_robustness_experiment_results(checkpoint_name, phase, exp):
    predictions = np.load(os.path.join(
        PATH, exp, f"robustness_{checkpoint_name}_predictions_{phase}.npy"))

    labels = np.load(os.path.join(
        PATH, exp, f"robustness_{checkpoint_name}_labels_{phase}.npy"))

    ori = softmax(predictions[:, 0, :]).mean(1)[:, 1]
    image = softmax(predictions[:, 1, :]).mean(1)[:, 1]
    text = softmax(predictions[:, 2, :]).mean(1)[:, 1]
    image_correspondence = softmax(predictions[:, 3:3+20, :]).mean(2)[:, :, 1]
    text_correspondence = softmax(predictions[:, 3+20:, :]).mean(2)[:, :, 1]

    return labels, ori, image, text, image_correspondence, text_correspondence

def visuals_per_model(outcomes, save_folder, checkpoint_name):
    # histogram_by_group(**outcomes)
    sns.set_theme(style="whitegrid")
    sns.set_context("talk")
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=False)

    violin_plot_by_group(axs[0], *outcomes)
    scatter_plot_instance_level(axs[1], *outcomes)
    fig.tight_layout()
    fig.savefig(os.path.join(save_folder, f"{checkpoint_name}.png"))

def epoch_wise_analysis(phase, exp, epochs, make_plots=True):

    results_auc = []
    results_corr = []

    for epoch in epochs:
        checkpoint_name = f'model_epoch_{epoch}'
        outcomes = load_robustness_experiment_results(checkpoint_name, phase, exp)

        # evaluate by AUC of all experiments
        aucs = AUC_table(*outcomes) # as a pd.DataFrame
        aucs['epoch'] = epoch
        results_auc.append(aucs) 

        # evaluate by correlation per group: image, text
        corr = get_correlation(*outcomes) # as a dict
        corr['epoch'] = epoch
        results_corr.append(corr)

        # do the visualizations of the correlations
        if make_plots:
            save_folder = os.path.join(
                'hatefulmeme', 'robustness_per_epoch', exp.replace('/', '_'), phase)
            if os.path.exists(save_folder) is False:
                os.makedirs(save_folder)
            
            visuals_per_model(outcomes, save_folder, checkpoint_name)
    
    # concatenate the results
    results_auc = pd.concat(results_auc, ignore_index=True)

    results_corr = pd.DataFrame(results_corr)
    results_corr.index = results_corr.epoch
    results_corr = results_corr.drop('epoch', axis=1)

    return results_auc, results_corr

from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_correlation_and_auc(results_corr, full, image, text):
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(hspace=0.15)

    outer = gridspec.GridSpec(2, 1, height_ratios = [4, 3]) 
    gs1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec = outer[0], hspace = .0)
    gs2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = outer[1])
    
    # adding axes to the GridSpec to assist the plotting
    axes = []
    for cell in gs1:
        axes.append(plt.subplot(cell))

    for cell in gs2:
        axes.append(plt.subplot(cell))

    axes[0].plot(results_corr.index, abs(results_corr.image), 'o--', label='image')
    axes[0].fill_between(results_corr.index, 0, abs(results_corr.image), alpha=0.5)
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    axes[1].plot(results_corr.index, abs(results_corr.text), 'o--', color='orange', label='text')
    axes[1].fill_between(results_corr.index, 0, abs(results_corr.text), color='orange', alpha=0.5)
    axes[1].set_ylim(0, 1)
    axes[1].invert_yaxis() 
    axes[1].legend()

    axes[0].yaxis.set_major_locator(LinearLocator(3))
    axes[1].yaxis.set_major_locator(LinearLocator(3))

    axes[0].set_ylabel("|Pearson's R|")
    axes[1].set_ylabel("|Pearson's R|")

    axes[2].plot(results_corr.index, full, '*--', color='gray', label='image+text', alpha=0.8)
    axes[2].plot(results_corr.index, image, '*--', color=sns.color_palette()[0], label='image', alpha=0.8)
    axes[2].plot(results_corr.index, text, '*--', color='orange', label='text', alpha=0.8,)

    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel("AUROC")
    axes[2].yaxis.set_major_locator(LinearLocator(4))

    plt.legend(ncol=3, loc='lower center')
    plt.show()

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

def ensemble_overtime(epoches_to_ensemble):
    predictions = []
    for epoch in epoches_to_ensemble:
        checkpoint_name = f'model_epoch_{epoch}'
        labels, ori, image, text, image_correspondence, text_correspondence = \
            load_robustness_experiment_results(checkpoint_name, phase, exp)
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
