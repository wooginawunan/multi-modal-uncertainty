
# %% Analyze the robustness 
from dotenv import load_dotenv
load_dotenv('../env.sh')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy.stats import pearsonr


import seaborn as sns
sns.set_theme(style="whitegrid")
from matplotlib.colors import to_rgb
from matplotlib.collections import PolyCollection

# Environment variables
PATH = os.environ['RESULTS_DIR']

def softmax(x):
    return(np.exp(x)/np.exp(x).sum(-1, keepdims=True))


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

def visuals_per_model(outcomes, save_folder, checkpoint_name):
    # histogram_by_group(**outcomes)
    sns.set_theme(style="whitegrid")
    sns.set_context("talk")
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=False)

    violin_plot_by_group(axs[0], *outcomes)
    scatter_plot_instance_level(axs[1], *outcomes)
    fig.tight_layout()
    fig.savefig(os.path.join(save_folder, f"{checkpoint_name}.png"))
    fig.close()

def load_robustness_experiment_results(checkpoint_name, phase, exp, dataset):
    predictions = np.load(os.path.join(
        PATH, dataset, exp, f"robustness_{checkpoint_name}_predictions_{phase}.npy"))

    labels = np.load(os.path.join(
        PATH, dataset, exp, f"robustness_{checkpoint_name}_labels_{phase}.npy"))
    
    return predictions, labels

from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_correlation_and_performance(results_corr, full, image, text, y_label):
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
    axes[2].set_ylabel(y_label)
    axes[2].yaxis.set_major_locator(LinearLocator(4))

    plt.legend(ncol=3, loc='lower center')
    plt.show()


def draw_learning_curves(experiments, prefix, suffix, dataset, auc=True):
    all_dfs = []
    for exp in experiments:
        try:
        #    df = pd.read_csv(os.path.join(PATH, 'updated_encoder', exp, 'history.csv'))
            df = pd.read_csv(os.path.join(PATH, dataset, prefix, exp, suffix, 'history.csv'))
            df['model_type'] = exp
            all_dfs.append(df)
        except FileNotFoundError:
            print(f"File not found for {exp}")
        
    all_dfs = pd.concat(all_dfs)

    sns.set_theme(style="whitegrid")
    sns.set_context("paper", 
                    font_scale=1.5, 
                    rc={"lines.linewidth": 2.5})

    fig, axs = plt.subplots(3 if auc else 2, 3, figsize=(15, 8))
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

    if auc:
        sns.lineplot(x="epoch", y="val_auc",
                    hue="model_type",
                    ax = axs[2, 1],
                    data=all_dfs)

        sns.lineplot(x="epoch", y="test_auc",
                    hue="model_type",
                    ax = axs[2, 2],
                    data=all_dfs)

    plt.tight_layout()
    fig.suptitle(f'{dataset} {prefix} {suffix}')
    plt.savefig(f'{dataset}/learning_curves_{prefix.replace("/", "_")}_{suffix}.png')
    plt.show()
    return all_dfs