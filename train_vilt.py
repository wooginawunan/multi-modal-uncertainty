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

import torch.optim as optim

from transformers import ViltForImagesAndTextClassification

logger = logging.getLogger(__name__)

from src import dataset
from src.training_loop import _construct_default_callbacks
from src.framework import Model_
from src.utils import set_seed


def get_args(parser):
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--model_type", type=str, default="Vanilla", 
                        choices=["Vanilla", "MIMO-shuffle-instance", "MultiHead"])
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--dataset", type=str, choices=["food101", "hateful-meme-dataset"], default="hateful-meme-dataset")
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=40)

def add_conditional_args(args):
    datapath = os.path.join(os.environ['DATA_DIR'], args.dataset)
    if args.dataset == "food101":

        args.labels, _ = dataset.get_labels_and_frequencies(
                os.path.join(datapath, "train.jsonl")
            )
        args.n_classes = len(args.labels)

        args.auc = False
        args.error_cases_remover = False
        
    elif args.dataset == "hateful-meme-dataset":
        
        args.labels = list(range(2))
        args.n_classes = 2
        
        args.auc = True
        args.error_cases_remover = True

    return args
    
def acc(y_pred, y_true, eval):
    _, y_pred = y_pred.max(1)
    acc_pred = (y_pred == y_true).float().mean()
    return acc_pred * 100

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args

    args = add_conditional_args(args)

    set_seed(args.seed)

    print(args)
    train, valid, test = dataset.get_dataset_vilt(
        args, os.path.join(os.environ['DATA_DIR'], args.dataset))

    model = ViltForImagesAndTextClassification.from_pretrained(
        "dandelin/vilt-b32-mlm", 
        num_labels=args.n_classes, 
        num_images=1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )
    args.scheduler_metric = 'val_acc'

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
        data_forming_func=None,
        metrics=[acc],
        verbose=True,
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
                        scheduler_step_on="epoch",
                        auc=args.auc,
                        vilt=True,
                        args=args
                        )
    
"""

bsub -n 20 -gpu "num=1:mode=shared:j_exclusive=yes" python train_vilt.py --use_gpu --device 0 \
--save_path $RESULTS_DIR/food101/vilt/Vanilla/32_3e-5 \
--lr 3e-5 --batch_size 4 --dataset food101  

bsub -q gpu32 -n 20 -gpu "num=1:mode=shared:j_exclusive=yes" python train_vilt.py --use_gpu --device 0 \
--save_path $RESULTS_DIR/hateful-meme/vilt/Vanilla/32_3e-5 \
--lr 3e-5 --batch_size 4 --dataset hateful-meme-dataset 


"""