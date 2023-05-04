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
import torch.optim as optim

from transformers import ViltForImagesAndTextClassification
from transformers.optimization import get_cosine_schedule_with_warmup
from pytorch_pretrained_bert import BertAdam

logger = logging.getLogger(__name__)

from src import dataset
from src.training_loop import _construct_default_callbacks
from src.framework import Model_
from src.utils import set_seed
from src.model import (
    FlavaFusionTransfomer, 
    FlavaFusionTransfomerwithCLSToken
)
from src.mmbt import MultimodalBertClf


def get_args(parser):
    # general args
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--resume", action='store_true')

    # vanilla optimizer args
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10) # early stopping based training acc
    
    # dataset args
    parser.add_argument("--dataset", type=str, choices=["food101", "hateful-meme-dataset"], default="hateful-meme-dataset")
    parser.add_argument("--sample_size", type=int, default=None)
    
    # model args
    parser.add_argument("--framework", type=str, choices=["vilt", "flava", "mmbt"])
    parser.add_argument("--model_type", type=str, default="Vanilla", 
                        choices=["Vanilla", "MIMO-shuffle-instance", "MultiHead"])
    
    # flava mm encoder args
    parser.add_argument("--multimodal_num_attention_heads", type=int, default=3)
    parser.add_argument("--multimodal_num_hidden_layers", type=int, default=3)
    parser.add_argument("--clstoken", action='store_true')
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--avg_pool", action='store_true')

    # flava optimizer args
    parser.add_argument("--wd", type=int, default=0.001)

    # vilt/mmbt scheduler args
    parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=40)

    # mmbt args
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased", choices=["bert-base-uncased", "bert-large-uncased"])
    parser.add_argument("--drop_img_percent", type=float, default=0.0)
    # parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--embed_sz", type=int, default=300)
    parser.add_argument("--freeze_img", type=int, default=3)
    parser.add_argument("--freeze_txt", type=int, default=5)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--include_bn", type=int, default=True)

    # parser.add_argument("--lr", type=float, default=5e-5)
    # parser.add_argument("--lr_factor", type=float, default=0.5)
    # parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--n_workers", type=int, default=0)
    parser.add_argument("--num_image_embeds", type=int, default=3)
    parser.add_argument("--warmup", type=float, default=0.1)

def add_conditional_args(args):
    args.datapath = os.path.join(os.environ['DATA_DIR'], args.dataset)
    if args.dataset == "food101":

        args.labels, _ = dataset.get_labels_and_frequencies(
                os.path.join(args.datapath, "train.jsonl")
            )
        args.n_classes = len(args.labels)
        args.auc = False
        args.error_cases_remover = False

        args.name_extractor = lambda x: x.split('.')[0]
        
    elif args.dataset == "hateful-meme-dataset":
        
        args.labels = list(range(2))
        args.n_classes = 2
        args.auc = True
        args.error_cases_remover = True

        args.name_extractor = lambda x: x.split('/')[-1].split('.')[0]

    if args.avg_pool:
        assert args.model_type != "Vanilla", "avg_pool is NOT supported for Vanilla model"

    return args
    
def acc(y_pred, y_true, eval, dummy_dim=False):
    
    if dummy_dim:
        if not eval:
            y_pred = y_pred.view(-1, y_pred.shape[2])
            y_true = y_true.view(-1)
        else:
            y_pred = y_pred.mean(1)
        
    _, y_pred = y_pred.max(1)
    acc_pred = (y_pred == y_true).float().mean()
    return acc_pred * 100

def setup_mmbt(args):
    assert args.model_type == "Vanilla", "MMBT supports only Vanilla mode"

    model = MultimodalBertClf(args)
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
        t_total=args.total_steps,
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )

    args.scheduler_metric = 'val_acc'
    args.scheduler_step_on = 'epoch'

    def data_forming_func(x, y, phase="train"):
        return x, y

    args.data_forming_func = data_forming_func
    args.metrics = [acc]

    return args, model, optimizer, scheduler

def setup_vilt(args):
    assert args.model_type == "Vanilla", "Vilt supports only Vanilla mode"
    model = ViltForImagesAndTextClassification.from_pretrained(
            "dandelin/vilt-b32-mlm", 
            num_labels=args.n_classes, 
            num_images=1)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )
    args.scheduler_metric = 'val_acc'
    args.scheduler_step_on = 'epoch'
    
    args.data_forming_func = None
    args.metrics = [acc]

    return args, model, optimizer, scheduler

def setup_flava(args):
    model_cls = FlavaFusionTransfomerwithCLSToken if args.clstoken else \
            FlavaFusionTransfomer

    model = model_cls(out_dim=1 if args.model_type == "Vanilla" else 2,                  
            num_classes=args.n_classes,
            multimodal_num_attention_heads=args.multimodal_num_attention_heads,
            multimodal_num_hidden_layers=args.multimodal_num_hidden_layers,
            drop=args.dropout,
            avg_pool=args.avg_pool
            )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1.0e-9,
        weight_decay=args.wd,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train)*3,
        num_training_steps=len(train)*args.n_epochs,
    )
    args.scheduler_step_on = 'batch'
    args.scheduler_metric = None

    args.data_forming_func = partial(
        dataset.data_forming_func_transformer, 
        model_type=args.model_type)
    
    args.metrics = [acc]
    
    return args, model, optimizer, scheduler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args

    args = add_conditional_args(args)

    set_seed(args.seed)

    print(args)

    if args.framework == "mmbt":
        assert args.dataset=="food101", "MMBT is only supported for food101"
        
        train, valid, test, n_classes, vocab = dataset.get_food101(
            datapath=args.datapath, 
            batch_size=args.batch_size,
            drop_img_percent = args.drop_img_percent,
            max_seq_len = args.max_seq_len, 
            num_image_embeds = args.num_image_embeds,
            n_workers = args.n_workers)
        
        args.n_classes = n_classes
        args.vocab = vocab
        args.total_steps = len(train) / args.gradient_accumulation_steps * args.n_epochs

    else:
        if args.framework == "vilt":
            data_func = dataset.get_dataset_vilt
        elif args.framework == "flava":
            data_func = dataset.get_dataset_flava
        
        train, valid, test = data_func(args, args.datapath)

    if args.framework == "vilt":
        args, model, optimizer, scheduler = setup_vilt(args)

    elif args.framework == "flava":
        args, model, optimizer, scheduler = setup_flava(args)
    
    elif args.framework == "mmbt":
        args, model, optimizer, scheduler = setup_mmbt(args)
        

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
        data_forming_func=args.data_forming_func,
        metrics=args.metrics,
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
                        epochs=args.n_epochs, 
                        callbacks=callbacks,
                        patience=args.patience,
                        epoch_start=epoch_start,
                        scheduler_step_on=args.scheduler_step_on,
                        auc=args.auc,
                        vilt=args.framework == "vilt",
                        mmbt=args.framework == "mmbt",
                        freeze_img=args.freeze_img,
                        freeze_txt=args.freeze_txt,
                        gradient_accumulation_steps=args.gradient_accumulation_steps,
                        scheduler_metric=args.scheduler_metric,
                        )
    
"""

bsub -n 20 -gpu "num=1:mode=shared:j_exclusive=yes" python train_vilt.py --use_gpu --device 0 \
--save_path $RESULTS_DIR/food101/vilt/Vanilla/32_3e-5 \
--lr 3e-5 --batch_size 4 --dataset food101  

bsub -q gpu32 -n 20 -gpu "num=1:mode=shared:j_exclusive=yes" python train_vilt.py --use_gpu --device 0 \
--save_path $RESULTS_DIR/hateful-meme/vilt/Vanilla/32_3e-5 \
--lr 3e-5 --batch_size 4 --dataset hateful-meme-dataset 


"""