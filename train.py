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
from pytorch_pretrained_bert import BertAdam

logger = logging.getLogger(__name__)

from src import dataset
from src.model import MIMOResNet, model_configure, MIMOTransfomer
from src.training_loop import _construct_default_callbacks
from src.framework import Model_

# %%
def get_args(parser):
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--wd", type=int, default=0.001)
    parser.add_argument("--momentum", type=int, default=0.9)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--model_type", type=str, default="Vanilla", 
                        choices=["Vanilla", "MIMO-shuffle-instance", "MIMO-shuffle-view", "MultiHead", 
                                 "MIMO-shuffle-all", "single-model-weight-sharing"])
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--multimodal_num_attention_heads", type=int, default=3)
    parser.add_argument("--multimodal_num_hidden_layers", type=int, default=3)
    parser.add_argument("--transformer", action='store_true')
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0)

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

    train, valid, _ = dataset.get_fmnist(
        datapath = os.environ['DATA_DIR'], 
        batch_size=args.batch_size,
        download = True, 
        shuffle = True,
        seed=args.seed)
    
    if args.transformer:
        total_steps = len(train) * args.n_epochs
        print("Total steps: ", total_steps)
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0,},
        ]
        optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=args.lr,
            warmup=args.warmup,
            t_total=total_steps,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "max", patience=10, verbose=True, factor=0.5
        )
        args.scheduler_metric = 'val_acc'

    else:

        optimizer = torch.optim.SGD(model.parameters(), 
            lr=args.lr, 
            weight_decay=args.wd, 
            momentum=args.momentum)
    
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
            mode='min', 
            factor=0.1, 
            patience=10, 
            verbose=True, 
            threshold=0.0001, 
            threshold_mode='rel', 
            cooldown=0, 
            min_lr=0, 
            eps=1e-08)
        
        args.scheduler_metric = 'val_loss'
    
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
        data_forming_func=partial(dataset.data_forming_func, 
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
                        test_generator=valid,
                        steps_per_epoch=len(train),
                        validation_steps=len(valid),
                        test_steps=len(valid),
                        epochs=args.n_epochs - 1, 
                        callbacks=callbacks,
                        patience=args.patience,
                        epoch_start=epoch_start,
                        scheduler_step_on="epoch",
                        auc=False,
                        args=args
                        )
    