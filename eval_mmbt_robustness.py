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
import tqdm
logger = logging.getLogger(__name__)

from src import dataset
from src.mmbt import MultimodalBertClf
from src.training_loop import _load_pretrained_model
from src.utils import torch_to

# %%
def get_args(parser):
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the model")
    parser.add_argument("--phase", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to load the model")
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--n_repeats", type=int, default=20, help="Number of times to repeat the random sampling")
    parser.add_argument("--dataset", type=str, choices=["food101", "hateful-meme-dataset"], default="hateful-meme-dataset")
    
    parser.add_argument("--num_image_embeds", type=int, default=3)
    parser.add_argument("--drop_img_percent", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--datapath", type=str, )
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased", choices=["bert-base-uncased", "bert-large-uncased"])
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--n_workers", type=int, default=0)

    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--include_bn", type=int, default=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args

    train, val, test, n_classes, vocab = dataset.get_food101(
        datapath=args.datapath, 
        batch_size=args.batch_size,
        drop_img_percent = args.drop_img_percent,
        max_seq_len = args.max_seq_len, 
        num_image_embeds = args.num_image_embeds,
        n_workers = args.n_workers)
    args.n_classes = n_classes
    args.vocab = vocab

    model = MultimodalBertClf(args)

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
        for _, (x, y) in tqdm.tqdm(enumerate(data[args.phase])):
            outputs = []
            # prediction with image and text
            x, y = torch_to(x, args.device), torch_to(y, args.device)
            outputs.append(model(*x))

            # prediction with only image
            outputs.append(model.forward_img_only(*x))

            # prediction with only text
            outputs.append(model.forward_txt_only(*x))

            for type in ["image", "text"]:
                for i in range(args.n_repeats):
                    # image-only correspondence 
                    outputs.append(model.forward_control(*x, type))

            y_hat = torch.stack(outputs, dim=1).cpu()
            preds.append(y_hat)
            labels.append(y.cpu())

    preds = torch.cat(preds, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()

    checkpoint_name = args.checkpoint_path.split('/')[-1].split('.')[0]
    np.save(os.path.join(args.save_path, f"robustness_{checkpoint_name}_predictions_{args.phase}.npy"), preds)
    np.save(os.path.join(args.save_path, f"robustness_{checkpoint_name}_labels_{args.phase}.npy"), labels)
   

    S, M, C =  preds.shape
    print('Gathered predictions of {} samples, {} variants, {} classes'.format(S, M, C))
    
    print('Gathered labels of {} samples'.format(len(labels)))
    
