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
import itertools
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
    parser.add_argument("--transformer", action='store_true')
    parser.add_argument("--multimodal_num_attention_heads", type=int, default=3)
    parser.add_argument("--multimodal_num_hidden_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Models")
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
    
    print('Loading Checkpoint from {}'.format(args.checkpoint_path))
    _load_pretrained_model(model, args.checkpoint_path)

    if args.use_gpu and torch.cuda.is_available(): 
        base_device = torch.device("cuda:{}".format(args.device))
        model.to(base_device)
        logger.info("Sending model to {}".format(base_device))
     
    model.eval()
    
    outputs = []
    labels = []
    with torch.no_grad():
        m = 4
        for i in range(m):
            y_hat = []
            for step, (x, y) in enumerate(valid):
                
                if args.model_type != "single-model-weight-sharing":
                    x, y = dataset.data_forming_func(x, y, 'eval', model_type=args.model_type)
                    x, y = x.to(args.device), y.to(args.device)
                    
                    x_ = torch.zeros(x.shape).to(args.device)
                    # for k, j in itertools.product(range(m), range(m)):
                    #     assert x_[:, k, :, :, :].equal(x_[:, j, :, :, :])
                    for j in range(m):
                        if j!=i:
                            x_[:, j, :, :, :] = x[:, j, :, :, :]
                            
                    y_ = model(x_)
                else:
                    b, m, c, h, w = x.shape
                    x_ = torch.zeros(b, m-1, c, h, w).to(args.device)

                    k = 0
                    for j in range(m):
                        if j!=i:
                            x_[:, k, :, :, :] = x[:, j, :, :, :]
                            k+=1
                    
                    x_, y = dataset.data_forming_func(x_, y, 'eval', model_type=args.model_type)
                    x_, y = x_.to(args.device), y.to(args.device)
                            
                    y_ = model(x_)
                    
                    y_ = y_.view(b, m-1, y_.shape[-1])
                
                y_hat.append(y_.data.cpu().numpy())
                
                if i==0: labels.append(y.data.cpu().numpy())
            
            outputs.append(np.concatenate(y_hat, axis=0)) 
            
    outputs = np.stack(outputs, axis=0)
    M_, S, M, C =  outputs.shape
    print('Gathered predictions of {} samples, {} views, {} dups, {} classes'.format(S, M_, M, C))
    
    labels = np.concatenate(labels, axis=0)
    print('Gathered labels of {} samples'.format(len(labels)))
    
    print('Saving predictions and labels to {}'.format(args.save_path))
    
    checkpoint_name = args.checkpoint_path.split('/')[-1].split('.')[0]
    np.save(os.path.join(args.save_path, 
                         f"{checkpoint_name}_predictions_robustness.npy"), outputs)
    np.save(os.path.join(args.save_path, f"{checkpoint_name}_labels.npy"), labels)