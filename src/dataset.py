# -*- coding: utf-8 -*-
"""
Simple data getters. Each returns iterator for train and dataset for test/valid
training: 60,000
testing: 10,000
20% val: 12,000 

"""
from functools import partial
import pandas as pd
import numpy as np
import os
import logging
logging.getLogger(__name__)

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import datasets, transforms 
from torchvision.transforms import functional as F_trans


def data_forming_func_hateful_memes(x, y, phase, model_type):
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
    else:
        raise NotImplementedError
        
    return x, y


class QuarterCrop(object):
    """
    Input size is 28*28 for fasion MNIST    
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

class FlavaEncodedHatefulMemeDataset(Dataset):
    def __init__(self, predix_dir, phase):

        self.meta_data = pd.read_json(os.path.join(predix_dir, f"{phase}.jsonl"), lines=True)

        self.emb_dir = os.path.join(predix_dir, 'flava_embeds')

        print(f"Loaded {len(self.meta_data)} samples from {phase} set.")
        with open(os.path.join(predix_dir, 'flava_embeds', f'{phase}_error_cases.txt'), 'r') as f:
            error_cases = [int(x) for x in f.read().split('\n')[:-1]]
        self.meta_data = self.meta_data.drop(labels=error_cases, axis=0)

        print(f"Loaded {len(self.meta_data)} samples from {phase} set after removing {len(error_cases)} error cases.")

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        save_name = self.meta_data.iloc[idx]['img'].split('/')[-1].split('.')[0]
        label = self.meta_data.iloc[idx]['label']

        img_path = os.path.join(self.emb_dir, save_name+'.img')
        text_path = os.path.join(self.emb_dir, save_name+'.text')
        image_emb, text_emb = torch.load(img_path), torch.load(text_path)

        return image_emb, text_emb, label

def collate_fn_pad(batch):
    imgs, txts, labels = [], [], []
    for i, t, l in batch:
        imgs.append(i)
        txts.append(t)
        labels.append(l)

    imgs = pad_sequence(imgs, batch_first=True, padding_value=0.)
    txts = pad_sequence(txts, batch_first=True, padding_value=0.)
    labels = torch.tensor(labels)
    return (imgs, txts), labels

def get_hatefulmeme(
        datapath = os.environ['DATA_DIR'], 
        batch_size = 128,
        shuffle = True,
        sample_size = None,
        seed=777):

    print(datapath)

    torch.manual_seed(seed)
    training = FlavaEncodedHatefulMemeDataset(datapath, 'train')
    dev = FlavaEncodedHatefulMemeDataset(datapath, 'dev')
    testing = FlavaEncodedHatefulMemeDataset(datapath, 'test')

    num_train = len(training)
    indices = list(range(num_train))
    training_idx = indices
    if sample_size is None:
        sample_size = len(training)
    training_idx = training_idx[:sample_size]

    training_sub = torch.utils.data.Subset(training, training_idx)

    training_loader = torch.utils.data.DataLoader(training_sub,
                                                   #sampler = torch.utils.data.SubsetRandomSampler(),
                                                   batch_size=batch_size,
                                                   shuffle=shuffle
                                                   ) 

    training_loader = torch.utils.data.DataLoader(training,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle,
                                                   collate_fn=collate_fn_pad
                                                   ) 

    dev_loader = torch.utils.data.DataLoader(dev,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            collate_fn=collate_fn_pad)
    
    test_loader = torch.utils.data.DataLoader(testing,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            collate_fn=collate_fn_pad)

    print('training_loader LENGTH:', len(training_loader))
        
    return training_loader, dev_loader, test_loader