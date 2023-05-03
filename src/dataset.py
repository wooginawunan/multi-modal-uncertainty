# -*- coding: utf-8 -*-
"""
Dataset for involved datasets:
- Hateful Memes Challenge
- Fashion MNIST
- Food101

"""
import pandas as pd
import numpy as np
import os
import logging
from collections import Counter
from functools import partial
logging.getLogger(__name__)

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import datasets, transforms 
from torchvision.transforms import functional as F_trans
from pytorch_pretrained_bert import BertTokenizer

from transformers import ViltProcessor

import json
from PIL import Image
from .utils import  numpy_seed

def data_forming_func_transformer(x, y, phase, model_type):
    """ transform the batch data according to the model type 
        - Vanilla: (B, 1) -> (B, 1)
        - MultiHead: (B, 1) -> (B, num_view)
        - MIMO-shuffle-instance: (B, num_view) -> (B, num_view) (shuffle the instance per view)
    """
    img, txt = x
    if model_type=='Vanilla' and phase=='train':
        y = y.unsqueeze(1).repeat(1, 1)
    
    elif model_type=="MultiHead" and phase=='train':
        y = y.unsqueeze(1).repeat(1, 2)
    
    elif model_type=="MIMO-shuffle-instance" and phase=='train':
        idx = torch.randperm(img.size(0))
        img = img[idx]
        y_img = y[idx]

        idx = torch.randperm(img.size(0))
        txt = txt[idx]
        y_txt = y[idx]
        
        y = torch.stack([y_img, y_txt], dim=1)
    
    return (img, txt), y

def data_forming_func(x, y, phase, model_type):

    b, m, c, h, w = x.shape
    if model_type=='Vanilla' and phase=='train':
        y = y.unsqueeze(1).repeat(1, 1)
    
    elif model_type=="single-model-weight-sharing":
        y = y.unsqueeze(1).repeat(1, m) # B, 4
        y = y.view(-1) # B*4
        x = x.view(-1, c, h, w)# B, 4, 1, 14, 14

    elif model_type=="MultiHead" and phase=='train':
        y = y.unsqueeze(1).repeat(1, m)
        
    elif model_type=="MIMO-shuffle-instance" and phase=='train':
        # x: B, 4, 1, 14, 14
        x_new = []
        y_new = []
        for i in range(4):
            idx = torch.randperm(x.size(0))
            x_new.append(x[idx, i, :, :, :])
            y_new.append(y[idx])
        
        x = torch.stack(x_new, dim=1)
        y = torch.stack(y_new, dim=1)      
        
    elif model_type=="MIMO-shuffle-view" and phase=='train':
        x = x[:, torch.randperm(x.size(1)), :, :, :]
        y = y.unsqueeze(1).repeat(1, m)
    
    elif model_type=="MIMO-shuffle-all" and phase=='train':
        x_new = []
        y_new = []
        for i in range(m):
            idx = torch.randperm(x.size(0))
            x_new.append(x[idx, i, :, :, :])
            y_new.append(y[idx])
        
        x = torch.stack(x_new, dim=1)
        y = torch.stack(y_new, dim=1) 
        
        ind =  torch.randperm(x.size(1))
        x = x[:, ind, :, :, :]
        y = y[:, ind]
        
    return x, y


class QuarterCrop(object):
    """
    Input size is 28*28 for fashion MNIST    
    """

    def __init__(self, expected_size):
        self.expected_size = expected_size
        self.crop_size_w = int(self.expected_size[0]/2)
        self.crop_size_h = int(self.expected_size[1]/2)

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        w, h = img.size
        assert w == self.expected_size[0] and h == self.expected_size[1]
        starts_x = [0, 0, self.crop_size_w, self.crop_size_h]
        starts_y = [0, self.crop_size_h, 0, self.crop_size_h]
        
        return [F_trans.crop(img, i, j, self.crop_size_h, self.crop_size_w) for i, j in zip(starts_x, starts_y)]


def get_fmnist(
        datapath = os.environ['DATA_DIR'], 
        batch_size=128,
        download = False, 
        shuffle = True,
        sample_size = None,
        seed=777):
    '''
    if in_quarter is True, img in batch will be cropped into four quarters,
    and the shape of each batch will be [batch_size, 4, 1, 14, 14], 
        0 - upper left
        1 - upper right
        2 - lower left
        3 - lower right
    No other data augmentation is applied.
    '''
    print(datapath)

    transform_to_quarter = transforms.Compose([
        QuarterCrop((28, 28)), 
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
    ])

    torch.manual_seed(seed)
    training = datasets.FashionMNIST(datapath, 
                                 download=download,
                                 train=True,
                                 transform=transform_to_quarter)
    
    testing = datasets.FashionMNIST(datapath,
                                    download=download,
                                    train=False,
                                    transform=transform_to_quarter)
    
    training_loader = torch.utils.data.DataLoader(training,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle
                                                   ) 
    
    test_loader = torch.utils.data.DataLoader(testing,
                                            batch_size=batch_size,
                                            shuffle=False)

    print('training_loader LENGTH:', len(training_loader))
        
    return training_loader, test_loader, None

class BaseDataset(Dataset):
    def __init__(self, predix_dir, phase,
                 label_dict=None, error_cases_remover=True, **kwargs):

        self.meta_data = pd.read_json(os.path.join(predix_dir, f"{phase}.jsonl"), lines=True)
        self.label_dict = label_dict

        print(f"Loaded {len(self.meta_data)} samples from {phase} set.")

        if error_cases_remover:
            with open(os.path.join(predix_dir, 'flava_embeds', f'{phase}_error_cases.txt'), 'r') as f:
                error_cases = [int(x) for x in f.read().split('\n')[:-1]]
            self.meta_data = self.meta_data.drop(labels=error_cases, axis=0)

            print(f"Loaded {len(self.meta_data)} samples from {phase} set after removing {len(error_cases)} error cases.")
    
    def __len__(self):
        return len(self.meta_data)

class FlavaEncodedDataset(BaseDataset):
    def __init__(self, predix_dir, phase,
                 label_dict, error_cases_remover=True, **kwargs):

        super().__init__(predix_dir, phase, label_dict, error_cases_remover, **kwargs)

        assert 'name_extractor' in kwargs
        self.name_extractor = kwargs['name_extractor']
        self.emb_dir = os.path.join(predix_dir, 'flava_embeds')
    
    def __getitem__(self, idx):
        save_name = self.name_extractor(self.meta_data.iloc[idx]['img'])
        img_path = os.path.join(self.emb_dir, save_name+'.img')
        text_path = os.path.join(self.emb_dir, save_name+'.text')
        label = self.meta_data.iloc[idx]['label']
        label = torch.LongTensor([self.label_dict.index(label)])
        
        return torch.load(img_path), torch.load(text_path), label
    

def collate_fn_flava(batch):
    imgs, txts, labels = [], [], []
    for i, t, l in batch:
        imgs.append(i)
        txts.append(t)
        labels.append(l)

    imgs = pad_sequence(imgs, batch_first=True, padding_value=0.)
    txts = pad_sequence(txts, batch_first=True, padding_value=0.)
    labels = torch.tensor(labels)
    return (imgs, txts), labels


class VILTDataset(BaseDataset):
    """HatefulMemeDataset."""

    def __init__(self, predix_dir, phase, 
                 label_dict, error_cases_remover=False, **kwargs):
        
        super().__init__(predix_dir, phase, label_dict, error_cases_remover, **kwargs)

        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        self.data_path = predix_dir

    def __getitem__(self, idx):
        # get image + text
        image_path, text = self.meta_data.iloc[idx][['img', 'text']]
        image = Image.open(os.path.join(self.data_path, image_path))
        
        encoding = self.processor(image, 
                                  text, 
                                  padding="max_length", 
                                  truncation=True, 
                                  return_tensors="pt")
        
        # remove batch dimension
        for k,v in encoding.items():
            encoding[k] = v.squeeze()

        # add labels
        encoding["labels"] = torch.LongTensor([
            self.label_dict.index(
                self.meta_data.iloc[idx]['label']
            )])

        return encoding


def collate_fn_vilt(batch):
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    input_ids = [item['input_ids'] for item in batch]
    pixel_values = [item['pixel_values'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # create padded pixel values and corresponding pixel mask
    encoding = processor.feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
    
    # create new batch
    batch = {}
    batch['input_ids'] = torch.stack(input_ids)
    batch['attention_mask'] = torch.stack(attention_mask)
    batch['token_type_ids'] = torch.stack(token_type_ids)
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask'].unsqueeze(1) # add dummy num_images dimensiono
    batch['labels'] = torch.stack(labels)
    
    return batch
    

def get_dataset(training, dev, testing, collate_func, args):

    torch.manual_seed(args.seed)
    num_train = len(training)
    indices = list(range(num_train))
    training_idx = indices

    if args.sample_size is None: 
        sample_size = len(training)
    else:
        sample_size = args.sample_size

    training_idx = training_idx[:sample_size]

    training_sub = torch.utils.data.Subset(training, training_idx)

    training_loader = torch.utils.data.DataLoader(training_sub,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   collate_fn=collate_func
                                                   ) 

    dev_loader = torch.utils.data.DataLoader(dev,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            collate_fn=collate_func)
    
    test_loader = torch.utils.data.DataLoader(testing,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            collate_fn=collate_func)

    print('training_loader LENGTH:', len(training_loader))
        
    return training_loader, dev_loader, test_loader


def get_dataset_flava(args, datapath):

    training = FlavaEncodedDataset(
        datapath, 'train', args.labels, args.error_cases_remover, 
        name_extractor=args.name_extractor)
    dev = FlavaEncodedDataset(
        datapath, 'dev', args.labels, args.error_cases_remover, 
        name_extractor=args.name_extractor)
    testing = FlavaEncodedDataset(
        datapath, 'test', args.labels, args.error_cases_remover, 
        name_extractor=args.name_extractor)

    return get_dataset(training, dev, testing, collate_fn_flava, args)


def get_dataset_vilt(args, datapath):

    training = VILTDataset(datapath, 'train', args.labels, args.error_cases_remover)
    dev = VILTDataset(datapath, 'dev', args.labels, args.error_cases_remover)
    testing = VILTDataset(datapath, 'test', args.labels, args.error_cases_remover)

    return get_dataset(training, dev, testing, collate_fn_vilt, args)


class JsonlDataset(Dataset):
    def __init__(self, data_path, tokenizer, transforms, vocab, n_classes, 
                 drop_img_percent, max_seq_len, num_image_embeds, labels):
        self.data = [json.loads(l) for l in open(data_path)]
        self.data_dir = os.path.dirname(data_path)
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.n_classes = n_classes
        self.text_start_token = ["[SEP]"]
        self.labels = labels

        with numpy_seed(0):
            for row in self.data:
                if np.random.random() < drop_img_percent:
                    row["img"] = None

        self.max_seq_len = max_seq_len
        self.max_seq_len -= num_image_embeds
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = (
            self.text_start_token
            + self.tokenizer(self.data[index]["text"])[:(self.max_seq_len - 1)]
        )
        segment = torch.zeros(len(sentence))

        sentence = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence
            ]
        )

        label = torch.LongTensor(
                [self.labels.index(self.data[index]["label"])]
            )
            
        image = None

        if self.data[index]["img"]:
            image = Image.open(
                os.path.join(self.data_dir, self.data[index]["img"])
            ).convert("RGB")
        else:
            image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
        image = self.transforms(image)

        # The first SEP is part of Image Token.
        segment = segment[1:]
        sentence = sentence[1:]
        # The first segment (0) is of images.
        segment += 1

        return sentence, segment, image, label


def get_labels_and_frequencies(path):
    label_freqs = Counter()
    data_labels = [json.loads(line)["label"] for line in open(path)]
    if type(data_labels[0]) == list:
        for label_row in data_labels:
            label_freqs.update(label_row)
    else:
        label_freqs.update(data_labels)

    return list(label_freqs.keys()), label_freqs


def collate_fn(batch):
    lens = [len(row[0]) for row in batch]
    bsz, max_seq_len = len(batch), max(lens)

    mask_tensor = torch.zeros(bsz, max_seq_len).long()
    text_tensor = torch.zeros(bsz, max_seq_len).long()
    segment_tensor = torch.zeros(bsz, max_seq_len).long()

    img_tensor = torch.stack([row[2] for row in batch])

    tgt_tensor = torch.cat([row[3] for row in batch]).long()

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        tokens, segment = input_row[:2]
        text_tensor[i_batch, :length] = tokens
        segment_tensor[i_batch, :length] = segment
        mask_tensor[i_batch, :length] = 1

    return (text_tensor, segment_tensor, mask_tensor, img_tensor), tgt_tensor

class Vocab(object):
    def __init__(self, emptyInit=False):
        if emptyInit:
            self.stoi, self.itos, self.vocab_sz = {}, [], 0
        else:
            self.stoi = {
                w: i
                for i, w in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
            }
            self.itos = [w for w in self.stoi]
            self.vocab_sz = len(self.itos)

    def add(self, words):
        cnt = len(self.itos)
        for w in words:
            if w in self.stoi:
                continue
            self.stoi[w] = cnt
            self.itos.append(w)
            cnt += 1
        self.vocab_sz = len(self.itos)

def get_vocab(bert_model):
    vocab = Vocab()

    bert_tokenizer = BertTokenizer.from_pretrained(
            bert_model, do_lower_case=True
        )
    vocab.stoi = bert_tokenizer.vocab
    vocab.itos = bert_tokenizer.ids_to_tokens
    vocab.vocab_sz = len(vocab.itos)

    return vocab

def get_food101(
        bert_model = 'bert-base-uncased',
        datapath = os.environ['DATA_DIR'], 
        drop_img_percent = 0.0, 
        max_seq_len = 512, 
        num_image_embeds = 3,
        batch_size = 128,
        n_workers = 20,
        ):
    
    tokenizer = (
        BertTokenizer.from_pretrained(bert_model, do_lower_case=True).tokenize
    )

    transforms_composed = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.46777044, 0.44531429, 0.40661017],
                std=[0.12221994, 0.12145835, 0.14380469],
            ),
        ]
    )

    labels, _ = get_labels_and_frequencies(
        os.path.join(datapath, "train.jsonl")
    )
    vocab = get_vocab(bert_model)
    n_classes = len(labels)

    train = JsonlDataset(
        os.path.join(datapath, "train.jsonl"), tokenizer, transforms_composed, vocab,
        n_classes, drop_img_percent, max_seq_len, num_image_embeds, labels
    )

    dev = JsonlDataset(
        os.path.join(datapath, "dev.jsonl"), tokenizer, transforms_composed, vocab,
        n_classes, drop_img_percent, max_seq_len, num_image_embeds, labels
    )

    test_set = JsonlDataset(
        os.path.join(datapath, "test.jsonl"),tokenizer, transforms_composed, vocab,
        n_classes, drop_img_percent, max_seq_len, num_image_embeds, labels
    )

    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        collate_fn=collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        dev,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        collate_fn=collate_fn,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader, n_classes, vocab