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
import torch.optim as optim

logger = logging.getLogger(__name__)

from src import dataset
from src.mmbt import MultimodalBertClf
from src.training_loop import _construct_default_callbacks
from src.framework import Model_
from src.utils import set_seed


def get_args(parser):
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased", choices=["bert-base-uncased", "bert-large-uncased"])
    parser.add_argument("--datapath", type=str, )
    parser.add_argument("--drop_img_percent", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed_sz", type=int, default=300)
    parser.add_argument("--freeze_img", type=int, default=3)
    parser.add_argument("--freeze_txt", type=int, default=5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=40)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--n_workers", type=int, default=0)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--num_image_embeds", type=int, default=3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--resume", action='store_true')
    

def acc(y_pred, y_true, eval):
    _, y_pred = y_pred.max(1)
    acc_pred = (y_pred == y_true).float().mean()
    return acc_pred * 100

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args

    set_seed(args.seed)
    

    train, valid, test, n_classes, vocab = dataset.get_food101(
        datapath=args.datapath, 
        batch_size=args.batch_size,
        drop_img_percent = args.drop_img_percent,
        max_seq_len = args.max_seq_len, 
        num_image_embeds = args.num_image_embeds,
        n_workers = args.n_workers)
    args.n_classes = n_classes
    args.vocab = vocab

    model = MultimodalBertClf(args)

    total_steps = len(train) / args.gradient_accumulation_steps * args.n_epochs
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
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
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

    def data_forming_func(x, y, model_type, eval=False, phase="train"):

        if model_type=='Vanilla':
            return x, y
        elif model_type=='MultiHead':
            text_tensor, segment_tensor, mask_tensor, img_tensor = x
            tgt_tensor = y

            
        elif model_type=='MIMO-shuffle-instance':
            text_tensor, segment_tensor, mask_tensor, img_tensor, tgt_tensor = x
            tgt_tensor = y

        return x, y
        
    model = Model_(model=model, 
        optimizer=optimizer, 
        scheduler=scheduler,
        data_forming_func=data_forming_func,
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
                        auc=False,
                        mmbt=True,
                        args=args
                        )
    
    """
    bsub -q short -Is -n 20 -gpu "num=1:mode=shared:j_exclusive=yes" python train_food101_mmbt.py --use_gpu --device 0\
        --save_path /gpfs/data/geraslab/Nan/multi_modal_uncertainty/saves/food101/mmbt/1e_4_4\
        --datapath /gpfs/data/geraslab/Nan/multi_modal/food101/ 
    """