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
experiments = [ 'Vanilla', #'MultiHead', 'MIMO-shuffle-instance'
               ]
prefix, suffix = 'head3_layer3/clip_transformer', '128_0.01_save_all_checkpoints'
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

# Read in the data
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
    
    # ax.set_ylim([-1, 1])
    # ax.set_xlim([-1, 1])

    ax.plot([data['experimental'].min(), data['experimental'].max()], 
              [data['experimental'].min(),data['experimental'].max()],
                'k--', 
              color='black', alpha=0.5)
    # ax.vlines(0, ymin=-1, ymax=1, colors='black', alpha=0.5)
    # ax.hlines(0, xmin=-1, xmax=1, colors='black', alpha=0.5)
    
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

# %%
checkpoint_name = 'model_best_val'
phase = 'test'
for exp in ['head3_layer3/clip_transformer/Vanilla/128_0.1',
            'head3_layer3/clip_transformer/Vanilla/32_0.01',
            'head3_layer3/clip_transformer/Vanilla/128_0.01',
            ]:


    predictions = np.load(os.path.join(
        PATH, exp, f"robustness_{checkpoint_name}_predictions_{phase}.npy"))

    labels = np.load(os.path.join(
        PATH, exp, f"robustness_{checkpoint_name}_labels_{phase}.npy"))

    ori = softmax(predictions[:, 0, :]).mean(1)[:, 1]
    image = softmax(predictions[:, 1, :]).mean(1)[:, 1]
    text = softmax(predictions[:, 2, :]).mean(1)[:, 1]
    image_correspondence = softmax(predictions[:, 3:3+20, :]).mean(2)[:, :, 1]
    text_correspondence = softmax(predictions[:, 3+20:, :]).mean(2)[:, :, 1]

    aucs = AUC_table(labels, ori, image, text, image_correspondence, text_correspondence)
    aucs['experiment'] = exp

    #histogram_by_group(labels, ori, image, text, image_correspondence, text_correspondence)
    sns.set_theme(style="whitegrid")
    sns.set_context("talk")
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=False)
    violin_plot_by_group(axs[0], labels, ori, image, text, image_correspondence, text_correspondence)
    scatter_plot_instance_level(axs[1], labels, ori, image, text, image_correspondence, text_correspondence)
    fig.tight_layout()
    fig.show()
    fig.savefig(
        os.path.join('hatefulmeme', f"robustness_{checkpoint_name}_violin_{phase}_{exp.replace('/', '_')}.pdf"))
    

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
# %%


# g.set(ylim=(-.3, 0.6))
# g.set(xlim=(-.4, 0.6))
# g.ax.xaxis.grid(True, "minor", linewidth=.25)
# g.ax.yaxis.grid(True, "minor", linewidth=.25)
# g.despine(left=True, bottom=True, right=True, top=True)


# # %% accuracy 

# ori = predictions[:, 0, :].argmax(-1)
# image = predictions[:, 1, :].argmax(-1)
# text = predictions[:, 2, :].argmax(-1)
# image_correspondence = predictions[:, 3:3+20, :].argmax(-1)
# text_correspondence = predictions[:, 3+20:, :].argmax(-1)

# print('accuracy with full inputs:', (ori==labels).mean())
# print('accuracy with image:', (image==labels).mean())
# print('accuracy (control group):', 
#       (image_correspondence==np.expand_dims(labels, 1)).mean())
# print('accuracy with text:', (text==labels).mean())
# print('accuracy (control group):', 
#       (text_correspondence==np.expand_dims(labels, 1)).mean())

# %% Timewise analysis


# %%

phase = 'test'

exp = 'head3_layer3/clip_transformer/Vanilla/128_0.1_save_all_checkpoints'

results = []
for epoch in range(1, 70):
    checkpoint_name = f'model_epoch_{epoch}'

    predictions = np.load(os.path.join(
        PATH, exp, f"robustness_{checkpoint_name}_predictions_{phase}.npy"))

    labels = np.load(os.path.join(
        PATH, exp, f"robustness_{checkpoint_name}_labels_{phase}.npy"))

    ori = softmax(predictions[:, 0, :]).mean(1)[:, 1]
    image = softmax(predictions[:, 1, :]).mean(1)[:, 1]
    text = softmax(predictions[:, 2, :]).mean(1)[:, 1]
    image_correspondence = softmax(predictions[:, 3:3+20, :]).mean(2)[:, :, 1]
    text_correspondence = softmax(predictions[:, 3+20:, :]).mean(2)[:, :, 1]

    aucs = AUC_table(labels, ori, image, text, image_correspondence, text_correspondence)
    aucs['epoch'] = epoch
    results.append(aucs)

    # histogram_by_group(labels, ori, image, text, image_correspondence, text_correspondence)
    # sns.set_theme(style="whitegrid")
    # sns.set_context("talk")
    # fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=False)
    # violin_plot_by_group(axs[0], labels, ori, image, text, image_correspondence, text_correspondence)
    # scatter_plot_instance_level(axs[1], labels, ori, image, text, image_correspondence, text_correspondence)
    # fig.tight_layout()
    # fig.show()
    # fig.savefig(
    #     os.path.join('hatefulmeme', f"robustness_{checkpoint_name}_violin_{phase}_{exp.replace('/', '_')}.pdf"))
    

results = pd.concat(results, ignore_index=True)


    
# %%
plt.figure(figsize=(12, 6))
sns.lineplot(data=results, x="epoch", y="AUC", hue="variants")
# %%
