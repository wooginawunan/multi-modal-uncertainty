# -*- coding: utf-8 -*-
"""
A gorgeous, self-contained, training loop. Uses Poutyne implementation, but this can be swapped later.
"""

import logging
import os
import pickle
from functools import partial

import numpy as np
import pandas as pd
import torch

from src.callbacks import ModelCheckpoint, LambdaCallback
from src.utils import save_weights

logger = logging.getLogger(__name__)

types_of_instance_to_save_in_csv = (int, float, complex, np.int64, np.int32, np.float32, np.float64, np.float128, str)
types_of_instance_to_save_in_history = (int, float, complex, np.int64, np.int32, np.float32, np.float64, np.ndarray, np.float128,str)

def _construct_default_callbacks(model, optimizer, H, save_path, checkpoint_monitor):
    callbacks = []
    callbacks.append(LambdaCallback(on_epoch_end=partial(_append_to_history_csv, H=H)))

    callbacks.append(
        LambdaCallback(
            on_epoch_end=partial(_save_history_csv, 
            save_path=save_path, 
            H=H)
        )
    )
    
    callbacks.append(ModelCheckpoint(monitor=checkpoint_monitor,
                                 save_best_only=True,
                                 mode='max',
                                 filepath=os.path.join(save_path, "model_best_val.pt")))
    
    def save_weights_fnc(epoch, logs):
        logger.info("Saving model from epoch " + str(epoch))
        save_weights(model, optimizer, os.path.join(save_path, "model_last_epoch.pt"))

    callbacks.append(LambdaCallback(on_epoch_end=save_weights_fnc))

    return callbacks


def _save_history_csv(epoch, logs, save_path, H):
    out = ""
    for key, value in logs.items():
        if isinstance(value, types_of_instance_to_save_in_csv):
            out += "{key}={value}\t".format(key=key, value=value)
    logger.info(out)
    logger.info("Saving history to " + os.path.join(save_path, "history.csv"))
    H_tosave = {}
    for key, value in H.items():
        if isinstance(value[-1], types_of_instance_to_save_in_csv):
            H_tosave[key] = value
    pd.DataFrame(H_tosave).to_csv(os.path.join(save_path, "history.csv"), index=False)


def _append_to_history_csv(epoch, logs, H):
    for key, value in logs.items():
        if key not in H:
            H[key] = [value]
        else:
            H[key].append(value)


def _load_pretrained_model(model, save_path):
    checkpoint = torch.load(save_path)
    model_dict = model.state_dict()
    model_dict.update(checkpoint['model']) 
    model.load_state_dict(model_dict, strict=True)
    logger.info("Done reloading!")
