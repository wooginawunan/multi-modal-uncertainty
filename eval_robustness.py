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
from src.model import MIMOResNet
from src.training_loop import _load_pretrained_model

# %%
def get_args(parser):
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to load the model")
    parser.add_argument("--model_type", type=str, default="Vanilla", 
                        choices=["Vanilla", "MIMO-shuffle-instance", "MIMO-shuffle-view", "MultiHead"])
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--batch_size", type=int, default=64)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    
    if args.model_type == "Vanilla":
        out_dim = 1
    else:
        out_dim = 4
    
    model = MIMOResNet(num_channels=1, emb_dim=4, out_dim=out_dim, num_classes=10)
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
                x, y = x.to(args.device), y.to(args.device)
                
                x_ = torch.zeros(x.shape).to(args.device)
                # for k, j in itertools.product(range(m), range(m)):
                #     assert x_[:, k, :, :, :].equal(x_[:, j, :, :, :])
                for j in range(m):
                    if j!=i:
                        x_[:, j, :, :, :] = x[:, j, :, :, :]
                        
                y_ = model(x_)
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


"""
python eval_robustness.py --model_type MultiHead\
--verbose\
--save_path $RESULTS_DIR/MultiHead/0.01\
--use_gpu\
--device 0\
--checkpoint_path $RESULTS_DIR/MultiHead/0.01/model_best_val.pt
    
python eval_robustness.py --model_type MIMO-shuffle-view\
    --verbose\
    --save_path $RESULTS_DIR/MIMO-shuffle-view/0.01\
    --use_gpu\
    --device 0\
    --checkpoint_path $RESULTS_DIR/MIMO-shuffle-view/0.01/model_best_val.pt
    
python eval_robustness.py --model_type MIMO-shuffle-instance\
    --verbose\
    --save_path $RESULTS_DIR/MIMO_shuffle_instance/0.01\
    --use_gpu\
    --device 0\
    --checkpoint_path $RESULTS_DIR/MIMO_shuffle_instance/0.01/model_best_val.pt
    
python eval_robustness.py --model_type Vanilla\
    --verbose\
    --save_path $RESULTS_DIR/Vanilla/0.01\
    --use_gpu\
    --device 0\
    --checkpoint_path $RESULTS_DIR/Vanilla/0.01/model_best_val.pt
"""