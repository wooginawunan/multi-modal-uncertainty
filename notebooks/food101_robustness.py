# %%
from dotenv import load_dotenv
load_dotenv('../env.sh')

import pandas as pd
import seaborn as sns
import os
import numpy as np

import seaborn as sns

# Environment variables
PATH = os.environ['RESULTS_DIR']

sns.set_theme(style="whitegrid")
from utils import (softmax,
                    get_correlation, 
                    visuals_per_model,
                    load_robustness_experiment_results, 
                    plot_correlation_and_performance,
                    draw_learning_curves
                    )

def process_predictions_food101(predictions, labels, mmbt=False):
    ori = softmax(predictions[:, 0, :])
    image = softmax(predictions[:, 1, :])
    text = softmax(predictions[:, 2, :])
    image_correspondence = softmax(predictions[:, 3:3+20, :])
    text_correspondence = softmax(predictions[:, 3+20:, :])

    if not mmbt:
        ori = ori.mean(1)
        image = image.mean(1)
        text = text.mean(1)
        image_correspondence = image_correspondence.mean(2)
        text_correspondence = text_correspondence.mean(2)

    ori = np.array([ori[i, j] for i, j in enumerate(labels)])
    image = np.array([image[i, j] for i, j in enumerate(labels)])
    text = np.array([text[i, j] for i, j in enumerate(labels)])
    image_correspondence = np.array([image_correspondence[i, :, j] for i, j in enumerate(labels)])
    text_correspondence = np.array([text_correspondence[i, :, j] for i, j in enumerate(labels)])

    return labels, ori, image, text, image_correspondence, text_correspondence

def ACC_tabel(predictions, labels, mmbt=False):
    if mmbt:
        ori = predictions[:, 0, :].argmax(-1)
        image = predictions[:, 1, :].argmax(-1)
        text = predictions[:, 2, :].argmax(-1)
        image_correspondence = predictions[:, 3:3+20, :].argmax(-1)
        text_correspondence = predictions[:, 3+20, :].argmax(-1)
    
    else:
        ori = predictions[:, 0, :, :].mean(1).argmax(-1)
        image = predictions[:, 1, :, :].mean(1).argmax(-1)
        text = predictions[:, 2, :, :].mean(1).argmax(-1)
        image_correspondence = predictions[:, 3:3+20, :, :].mean(2).argmax(-1)
        text_correspondence = predictions[:, 3+20:, :, :].mean(2).argmax(-1)
            
    image_control = (image_correspondence==np.expand_dims(labels, 1)).mean(-1)
    text_control = (text_correspondence==np.expand_dims(labels, 1)).mean(-1)
        
    df =  pd.DataFrame({   
        'variants': ['full', 'image', 'text'],
        'ACC': [(ori==labels).mean()*100, 
                (image==labels).mean()*100, 
                (text==labels).mean()*100],
    })

    for i, auc in enumerate(image_control):
        df.loc[i + 3] = ['image_control', auc]
    for i, auc in enumerate(text_control):
        df.loc[i + 23] = ['text_control', auc]

    print(df.groupby('variants')['ACC'].agg(['mean', 'std']))
    return df


def epoch_wise_analysis(phase, exp, epochs, make_plots=True):

    dataset = 'food101'
    mmbt = 'mmbt' in exp

    results = []
    results_corr = []
    for epoch in epochs:
        checkpoint_name = f'model_epoch_{epoch}'
        try:
            predictions, labels = load_robustness_experiment_results(
                checkpoint_name, phase, exp, dataset)
        except FileNotFoundError:
            print(f'Checkpoint {checkpoint_name} not found')
            continue
            
        outcomes = process_predictions_food101(
            predictions, labels, mmbt)

        # performance
        df = ACC_tabel(predictions, labels, mmbt)
        df['epoch'] = epoch
        results.append(df) 

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
    
    if len(results)>0:
        # concatenate the results
        results = pd.concat(results, ignore_index=True)

        results_corr = pd.DataFrame(results_corr)
        results_corr.index = results_corr.epoch
        results_corr = results_corr.drop('epoch', axis=1)

    return results, results_corr


def run_per_experiment(phase, exp, epochs, make_plots):

    results, results_corr = epoch_wise_analysis(phase, exp, epochs, make_plots=make_plots)

    if len(results)>0:
        group_by_table = results.groupby(['variants', 'epoch']).mean().reset_index()
        full = group_by_table[group_by_table['variants']=='full'].ACC.values
        text = group_by_table[group_by_table['variants']=='text'].ACC.values
        image = group_by_table[group_by_table['variants']=='image'].ACC.values
        image_control = group_by_table[group_by_table['variants']=='image_control'].ACC.values
        text_control = group_by_table[group_by_table['variants']=='text_control'].ACC.values

        plot_correlation_and_performance(results_corr, full, image, text, 'Accuracy')
        sns.lineplot(data=results, hue='variants', x='epoch', y='ACC')

# %%


phase = 'val'
epochs = range(1, 100)

experiments = ('head3_layer3/clip_transformer/128_0.1',
                'head3_layer3/clip_transformer/128_0.001',
                'head3_layer3/clip_transformer/128_1e-5',
                
                'head3_layer3/clip_transformer/MIMO-shuffle-instance/128_0.1'
                'head3_layer3/clip_transformer/MIMO-shuffle-instance/128_0.001',

                'head6_layer6/clip_transformer/128_0.001',
                'mmbt/5e_5_4')
for exp in experiments:
    run_per_experiment(phase, exp, epochs, make_plots=True)



# %%
run_per_experiment(phase, 'head3_layer3/clip_transformer/128_1e-5', epochs, make_plots=True)
# %%
