# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import os
import torch
import numpy as np
import argparse
import logging
from functools import partial
logger = logging.getLogger(__name__)

from src import dataset
from src.model import FlavaFusionTransfomer
from src.training_loop import _load_pretrained_model
from src.utils import torch_to

# %%
def get_args(parser):
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the model")
    parser.add_argument("--phase", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to load the model")
    parser.add_argument("--model_type", type=str, default="Vanilla", 
                        choices=["Vanilla", "MIMO-shuffle-instance", "MultiHead"])
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--n_repeats", type=int, default=20, help="Number of times to repeat the random sampling")
    parser.add_argument("--multimodal_num_attention_heads", type=int, default=3)
    parser.add_argument("--multimodal_num_hidden_layers", type=int, default=3)

def input_sampling(l_img, l_txt, type="image"):
    """
    type: image or text
    """
    assert type in ["image", "text"]

    l = l_img if type == "image" else l_txt
    n = np.random.randint(0, l+1, size=1)[0]

    n_img = n if type == "image" else l-n
    n_txt = n if type == "text" else l-n
    
    indices_img, _ = torch.sort(torch.randperm(l_img)[:n_img])
    indices_txt, _ = torch.sort(torch.randperm(l_txt)[:n_txt])

    return indices_img, indices_txt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args

    if args.model_type == "Vanilla":
        model = FlavaFusionTransfomer(out_dim=1,                  
                multimodal_num_attention_heads=args.multimodal_num_attention_heads,
                multimodal_num_hidden_layers=args.multimodal_num_hidden_layers
                )
    elif args.model_type == "MIMO-shuffle-instance" or args.model_type == "MultiHead":
        model = FlavaFusionTransfomer(out_dim=2,
                multimodal_num_attention_heads=args.multimodal_num_attention_heads,
                multimodal_num_hidden_layers=args.multimodal_num_hidden_layers
                )

    train, val, test = dataset.get_hatefulmeme(
        datapath = os.environ['DATA_DIR'], 
        batch_size=args.batch_size,
        shuffle = True,
        seed=args.seed)

    data = {'train': train, 'val': val, 'test': test}

    _load_pretrained_model(model, args.checkpoint_path)

    if args.use_gpu and torch.cuda.is_available(): 
        base_device = torch.device("cuda:{}".format(args.device))
        model.to(base_device)
        logger.info("Sending model to {}".format(base_device))
            
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for _, (x, y) in enumerate(data[args.phase]):
            outputs = []
            # prediction with image and text
            x, y = torch_to(x, args.device), torch_to(y, args.device)
            outputs.append(model(x))

            img, txt = x
            l_img, l_txt = img.shape[1], txt.shape[1]

            # prediction with only image
            outputs.append(model((img, None)))

            # prediction with only text
            outputs.append(model((None, txt)))

            for type in ["image", "text"]:
                for i in range(args.n_repeats):
                    # image-only correspondence 
                    indices_img, indices_txt = input_sampling(l_img, l_txt, type)
                    s_img = img[:, indices_img, :] if len(indices_img)>0 else None
                    s_txt = img[:, indices_txt, :] if len(indices_txt)>0 else None

                    outputs.append(model((s_img, s_txt)))

            y_hat = torch.stack(outputs, dim=1).cpu()
            preds.append(y_hat)
            labels.append(y.cpu())

    preds = torch.cat(preds, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()

    S, M, K, C =  preds.shape
    print('Gathered predictions of {} samples, {} variants, {} heads, {} classes'.format(S, M, K, C))
    
    print('Gathered labels of {} samples'.format(len(labels)))
    
    checkpoint_name = args.checkpoint_path.split('/')[-1].split('.')[0]
    np.save(os.path.join(args.save_path, f"robustness_{checkpoint_name}_predictions_{args.phase}.npy"), preds)
    np.save(os.path.join(args.save_path, f"robustness_{checkpoint_name}_labels_{args.phase}.npy"), labels)
    
    """
    bsub -q short -Is -n 20 -gpu "num=1:mode=shared:j_exclusive=yes" python eval_transformer_robustness.py --model_type Vanilla\
        --verbose\
        --save_path $RESULTS_DIR/head3_layer3/clip_transformer/Vanilla/32_0.01\
        --use_gpu\
        --device 0\
        --checkpoint_path $RESULTS_DIR/head3_layer3/clip_transformer/Vanilla/32_0.01/model_last_epoch.pt\
        --batch_size 128\
        --phase val\
        --n_repeats 20
    bsub -q short -Is -n 20 -gpu "num=1:mode=shared:j_exclusive=yes"  python eval_transformer_robustness.py --model_type MIMO-shuffle-instance\
        --verbose\
        --save_path $RESULTS_DIR/head3_layer3/clip_transformer/MIMO-shuffle-instance/32_0.01\
        --use_gpu\
        --device 0\
        --checkpoint_path $RESULTS_DIR/head3_layer3/clip_transformer/MIMO-shuffle-instance/32_0.01/model_last_epoch.pt\
        --batch_size 128\
        --phase val\
        --n_repeats 20
    bsub -q short -Is -n 20 -gpu "num=1:mode=shared:j_exclusive=yes"  python eval_transformer_robustness.py --model_type MultiHead\
        --verbose\
        --save_path $RESULTS_DIR/head3_layer3/clip_transformer/MultiHead/32_0.01\
        --use_gpu\
        --device 0\
        --checkpoint_path $RESULTS_DIR/head3_layer3/clip_transformer/MultiHead/32_0.01/model_last_epoch.pt\
        --batch_size 128\
        --phase val\
        --n_repeats 20
    """