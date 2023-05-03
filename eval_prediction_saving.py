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
from src.model import MIMOResNet, model_configure, MIMOTransfomer
from src.training_loop import _load_pretrained_model

# %%
def get_args(parser):
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to load the model")
    parser.add_argument("--model_type", type=str, default="Vanilla", 
                        choices=["Vanilla", "MIMO-shuffle-instance", "MIMO-shuffle-view", "MultiHead", 
                                 "MIMO-shuffle-all", "single-model-weight-sharing"])
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--multimodal_num_attention_heads", type=int, default=3)
    parser.add_argument("--multimodal_num_hidden_layers", type=int, default=3)
    parser.add_argument("--transformer", action='store_true')
    parser.add_argument("--dropout", type=float, default=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    
    emb_dim, out_dim = model_configure[args.model_type]
    if args.transformer:
        assert args.model_type == "MultiHead" or args.model_type == "MIMO-shuffle-instance"
        model = MIMOTransfomer(
                    out_dim=out_dim, 
                    num_classes=10,
                    image_dim=14*14,
                    hidden_size=768,
                    multimodal_num_attention_heads=args.multimodal_num_attention_heads,
                    multimodal_num_hidden_layers=args.multimodal_num_hidden_layers,
                    drop=args.dropout,
                )
    else:
        model = MIMOResNet(
                num_channels=1, 
                emb_dim=emb_dim, 
                out_dim=out_dim, 
                num_classes=10
            )
    _, valid, _ = dataset.get_fmnist(
        datapath = os.environ['DATA_DIR'], 
        batch_size=args.batch_size,
        download = True, 
        shuffle = True,
        seed=args.seed)
    
    _load_pretrained_model(model, args.checkpoint_path)

    if args.use_gpu and torch.cuda.is_available(): 
        base_device = torch.device("cuda:{}".format(args.device))
        model.to(base_device)
        logger.info("Sending model to {}".format(base_device))
     
    model.eval()
    
    outputs = []
    labels = []
    with torch.no_grad():
        for step, (x, y) in enumerate(valid):
            b, m = x.shape[0], x.shape[1]
            x, y = dataset.data_forming_func(x, y, 'eval', model_type=args.model_type)
            x, y = x.to(args.device), y.to(args.device)
            y_hat = model(x)
            
            if args.model_type == "single-model-weight-sharing":
                y_hat = y_hat.view(b, m, y_hat.shape[-1])
                
            outputs.append(y_hat.data.cpu().numpy())
            
            if args.model_type == "single-model-weight-sharing":
                y = y.view(b, m)
                labels.append(y[:, 0].data.cpu().numpy())
            else:
                labels.append(y.data.cpu().numpy())
            
    outputs = np.concatenate(outputs, axis=0)
    S, M, C =  outputs.shape
    print('Gathered predictions of {} samples, {} views, {} classes'.format(S, M, C))
    
    labels = np.concatenate(labels, axis=0)
    print('Gathered labels of {} samples'.format(len(labels)))
    
    checkpoint_name = args.checkpoint_path.split('/')[-1].split('.')[0]
    np.save(os.path.join(args.save_path, f"{checkpoint_name}_predictions.npy"), outputs)
    np.save(os.path.join(args.save_path, f"{checkpoint_name}_labels.npy"), labels)


    """
     python eval_prediction_saving.py --model_type MultiHead\
        --verbose\
        --save_path $RESULTS_DIR/MultiHead/0.01\
        --use_gpu\
        --device 0\
        --checkpoint_path $RESULTS_DIR/MultiHead/0.01/model_best_val.pt
    """