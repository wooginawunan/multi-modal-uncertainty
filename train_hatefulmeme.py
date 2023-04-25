# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import os
import torch
import pandas as pd
import argparse
import logging
from functools import partial
logger = logging.getLogger(__name__)

from src import dataset
from src.model import FlavaFusionTransfomer
from src.training_loop import _construct_default_callbacks
from src.framework import Model_
from transformers.optimization import get_cosine_schedule_with_warmup

# %%
def get_args(parser):
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--wd", type=int, default=0.001)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--model_type", type=str, default="Vanilla", 
                        choices=["Vanilla", "MIMO-shuffle-instance", "MultiHead"])
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--resume", action='store_true')

def acc(y_pred, y_true, eval):
    
    if not eval:
        y_pred = y_pred.view(-1, y_pred.shape[2])
        y_true = y_true.view(-1)
    else:
        y_pred = y_pred.mean(1)
        
    _, y_pred = y_pred.max(1)
    
    acc_pred = (y_pred == y_true).float().mean()
    return acc_pred * 100

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args

    if args.model_type == "Vanilla":
        model = FlavaFusionTransfomer(out_dim=1)
    elif args.model_type == "MIMO-shuffle-instance" or args.model_type == "MultiHead":
        model = FlavaFusionTransfomer(out_dim=2)

    train, valid, test = dataset.get_hatefulmeme(
        datapath = os.environ['DATA_DIR'], 
        batch_size=args.batch_size,
        shuffle = True,
        seed=args.seed)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1.0e-9,
        weight_decay=args.wd,
    )

    steps = int(len(train)/args.batch_size)+1
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=steps*3,
        num_training_steps=steps*args.n_epochs,
    )
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    if args.resume:
        model_last_epoch_path = os.path.join(args.save_path, "model_last_epoch.pt")
        checkpoint = torch.load(model_last_epoch_path)
        model.load_state_dict(checkpoint['model'])
        
        history_csv_path = os.path.join(args.save_path, "history.csv")
        H = pd.read_csv(history_csv_path)
        H = {col: list(H[col].values) for col in H.columns if col != "Unnamed: 0"}
        
        epoch_start = len(H['epoch'])+1
    else:
        H = {}
        history_csv_path = os.path.join(args.save_path, "history.csv")

        logger.info("Removing {}".format(history_csv_path))
        os.system("rm " + history_csv_path)
        epoch_start = 1
        
    callbacks = _construct_default_callbacks(model, optimizer, H, args.save_path, 
                                             checkpoint_monitor="val_acc")
    
    # Configure callbacks
    for clbk in callbacks:
        clbk.set_save_path(args.save_path)
        clbk.set_model(model, ignore=False)  # TODO: Remove this trick
        clbk.set_optimizer(optimizer)

    model = Model_(model=model, 
        optimizer=optimizer, 
        scheduler=scheduler,
        data_forming_func=partial(dataset.data_forming_func_hateful_memes, 
                                  model_type=args.model_type),
        metrics=[acc],
        verbose=args.verbose,
        )
            
    for clbk in callbacks:
        clbk.set_model_pytoune(model)

    if args.use_gpu and torch.cuda.is_available(): 
        base_device = torch.device("cuda:{}".format(args.device))
        model.to(base_device)
        logger.info("Sending model to {}".format(base_device))
        
    _ = model.train_loop(train,
                        valid_generator=valid,
                        test_generator=test,
                        steps_per_epoch=len(train),
                        validation_steps=len(valid),
                        test_steps=len(test),
                        epochs=args.n_epochs - 1, 
                        callbacks=callbacks,
                        patience=args.patience,
                        epoch_start=epoch_start,
                        scheduler_step_on="batch",
                        auc=True
                        )
    
    """
    python train.py --model_type MIMO-shuffle-all --batch_size 32 --lr 0.01 --verbose 
ESULTS_DIR/MIMO_shuffle_all/0.1 --use_gpu --device 3
    """