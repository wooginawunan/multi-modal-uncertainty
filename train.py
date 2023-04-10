# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import os
import torch
import argparse
import logging
from functools import partial
logger = logging.getLogger(__name__)

from src import dataset
from src.model import MIMOResNet
from src.training_loop import _construct_default_callbacks
from src.framework import Model_

# %%
def get_args(parser):
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=int, default=0.001)
    parser.add_argument("--momentum", type=int, default=0.9)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--model_type", type=str, default="Vanilla", 
                        choices=["Vanilla", "MIMO-shuffle-instance", "MIMO-shuffle-view", "MultiHead"])
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--patience", type=int, default=10)

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
        out_dim = 1
    else:
        out_dim = 4
    
    model = MIMOResNet(num_channels=1, emb_dim=4, out_dim=out_dim, num_classes=10)
    train, valid, _ = dataset.get_fmnist(
        datapath = os.environ['DATA_DIR'], 
        batch_size=args.batch_size,
        download = True, 
        shuffle = True,
        seed=args.seed)

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
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    history_csv_path = os.path.join(args.save_path, "history.csv")

    logger.info("Removing {}".format(history_csv_path))
    os.system("rm " + history_csv_path)

    H = {}
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
                        )
    
    """
    python train.py --model_type MultiHead --batch_size 32 --lr 0.01 --verbose 
ESULTS_DIR/MultiHead_test --use_gpu
    """