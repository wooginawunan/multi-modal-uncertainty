# -*- coding: utf-8 -*-
"""
Callbacks implementation. Inspired by Keras.
"""

import timeit
import sys
import numpy as np
import logging
import itertools

from src.utils import save_weights

logger = logging.getLogger(__name__)        

class CallbackList:
    def __init__(self, callbacks=None):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model):
        for callback in self.callbacks:
            callback.set_model(model)
    
    def set_model_pytoune(self, model_pytoune):
        for callback in self.callbacks:
            callback.set_model_pytoune(model_pytoune)

    def on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_forward_begin(self, batch, data):
        for callback in self.callbacks:
            callback.on_forward_begin(batch, data)

    def on_backward_end(self, batch):
        for callback in self.callbacks:
            callback.on_backward_end(batch)

    def on_train_begin(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_val_batch_end(self, batch, logs):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_val_batch_end(batch, logs)

    def __iter__(self):
        return iter(self.callbacks)

class Callback(object):
    def __init__(self):
        pass

    def set_meta_data(self, meta_data):
        self.meta_data = meta_data

    def set_save_path(self, save_path):
        self.save_path = save_path

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_model(self, model, ignore=True):
        if ignore:
            return
        self.model = model
    
    def set_model_pytoune(self, model_pytoune):
        self.model_pytoune = model_pytoune

    def set_params(self, params):
        self.params = params

    def set_dataloader(self, data):
        self.data = data

    def get_dataloader(self):
        return self.data

    def get_meta_data(self):
        return self.meta_dataset_model_pytoune

    def get_optimizer(self):
        return self.optimizer

    def get_params(self):
        return self.params

    def get_model(self):
        return self.model

    def get_save_path(self):
        return self.save_path

    def on_epoch_begin(self, epoch, logs):
        pass

    def on_epoch_end(self, epoch, logs):
        pass

    def on_batch_begin(self, batch, logs):
        pass

    def on_batch_end(self, batch, logs):
        pass

    def on_forward_begin(self, batch, data):
        pass

    def on_backward_end(self, batch):
        pass

    def on_train_begin(self, logs):
        pass

    def on_train_end(self, logs):
        pass

    def on_val_batch_end(self, batch, logs):
        pass

class LambdaCallback(Callback):
    def __init__(self,
                 on_epoch_begin=None,
                 on_epoch_end=None,
                 on_batch_begin=None,
                 on_batch_end=None,
                 on_train_begin=None,
                 on_train_end=None):
        super(LambdaCallback, self).__init__()
        if on_epoch_begin is not None:
            self.on_epoch_begin = on_epoch_begin
        else:
            self.on_epoch_begin = lambda epoch, logs: None
        if on_epoch_end is not None:
            self.on_epoch_end = on_epoch_end
        else:
            self.on_epoch_end = lambda epoch, logs: None
        if on_batch_begin is not None:
            self.on_batch_begin = on_batch_begin
        else:
            self.on_batch_begin = lambda batch, logs: None
        if on_batch_end is not None:
            self.on_batch_end = on_batch_end
        else:
            self.on_batch_end = lambda batch, logs: None
        if on_train_begin is not None:
            self.on_train_begin = on_train_begin
        else:
            self.on_train_begin = lambda logs: None
        if on_train_end is not None:
            self.on_train_end = on_train_end
        else:
            self.on_train_end = lambda logs: None

class ModelCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False,
                 mode='auto', period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['model']
        del state['optimizer']
        return state

    def __setstate__(self, newstate):
        newstate['model'] = self.model
        newstate['optimizer'] = self.optimizer
        self.__dict__.update(newstate)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    logging.warning('Can save best model only with %s available, '
                                    'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch, self.monitor, self.best,
                                     current, self.filepath))
                        self.best = current
                        save_weights(self.model, self.optimizer, self.filepath)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch, self.filepath))
                    save_weights(self.model, self.optimizer, self.filepath)
 
class ProgressionCallback(Callback):
    def __init__(self, other_metrics = []):
         
        self.other_metrics = []
        for me in other_metrics:
            self.other_metrics.append(me)

    def on_train_begin(self, logs):
        self.metrics = ['loss'] + self.model_pytoune.metrics_names
        self.epochs = self.params['epochs']
        self.steps = self.params['steps']

    def on_epoch_begin(self, epoch, logs):
        self.step_times_sum = 0.
        self.epoch = epoch
        sys.stdout.write("\rEpoch %d/%d" % (self.epoch, self.epochs))
        sys.stdout.flush()

    def on_epoch_end(self, epoch, logs):
        epoch_total_time = logs['time']

        metrics_str = self._get_metrics_string(logs)
        iol_str = self._get_iol_string(logs)
        if self.steps is not None:
            print("\rEpoch %d/%d %.2fs/%.2fs: Step %d/%d: %s. %s" %
                  (self.epoch, self.epochs, epoch_total_time, timeit.default_timer()-logs['epoch_begin_time'], self.steps, self.steps, metrics_str, iol_str))

        else:
            print("\rEpoch %d/%d %.2fs/%.2fs: Step %d/%d: %s. %s" %
                  (self.epoch, self.epochs, epoch_total_time, timeit.default_timer()-logs['epoch_begin_time'], self.last_step, self.last_step, metrics_str, iol_str))

    def on_batch_end(self, batch, logs):
        self.step_times_sum += timeit.default_timer()-logs['batch_begin_time']

        metrics_str = self._get_metrics_string(logs)
        iol_str = self._get_iol_string(logs)
        #print(iol_str)
        times_mean = self.step_times_sum / batch
        if self.steps is not None:
            remaining_time = times_mean * (self.steps - batch)
            
            sys.stdout.write("\rEpoch %d/%d ETA %.2fs Step %d/%d: %s. %s" %
                             (self.epoch, self.epochs, remaining_time, batch, self.steps, metrics_str, iol_str))
            if 'cumsum_iol' in iol_str: sys.stdout.write("\n")
            sys.stdout.flush()
        else:
            sys.stdout.write("\rEpoch %d/%d %.2fs/step Step %d: %s. %s" %
                             (self.epoch, self.epochs, times_mean, batch, metrics_str, iol_str))
            sys.stdout.flush()
            self.last_step = batch

    def _get_metrics_string(self, logs):
        train_metrics_str_gen = ('{}: {:f}'.format(k, logs[k]) for k in self.metrics if logs.get(k) is not None)
        val_metrics_str_gen = ('{}: {:f}'.format('val_' + k, logs['val_' + k]) for k in self.metrics
                               if logs.get('val_' + k) is not None)
        return ', '.join(itertools.chain(train_metrics_str_gen, val_metrics_str_gen))

    def _get_iol_string(self, logs):
        str_gen = ['{}: {:f}'.format(k, logs[k]) for k in self.other_metrics if logs.get(k) is not None]
        #print(str_gen, '\n',[(k, logs[k]) for k in ['average_iol_current_epoch', 'average_iol']])
        return  ', '.join(str_gen)

class ValidationProgressionCallback(Callback):
    def __init__(self, 
                 phase,
                 metrics_names,
                 steps=None):
        self.params = {}
        self.params['steps'] = steps
        self.params['phase'] = phase 
        self.metrics = metrics_names

        super(ValidationProgressionCallback, self).__init__()

    def _get_metrics_string(self, logs):
        metrics_str_gen = ('{}: {:f}'.format(self.params['phase'] + '_' + k, logs[k]) for k in self.metrics
                               if logs.get(k) is not None)
        return ', '.join(metrics_str_gen)

    def on_batch_begin(self, batch, logs):
        if batch==1:
            self.step_times_sum = 0.
        
        self.steps = self.params['steps']

    def on_batch_end(self, batch, logs):
        self.step_times_sum += timeit.default_timer()-logs['batch_begin_time']

        metrics_str = self._get_metrics_string(logs)
        times_mean = self.step_times_sum / batch
        if self.steps is not None:
            remaining_time = times_mean * (self.steps - batch)
            
            sys.stdout.write("\r%s ETA %.2fs Step %d/%d: %s." %
                             (self.params['phase'], remaining_time, batch, self.steps, metrics_str))
            sys.stdout.flush()
        else:
            sys.stdout.write("\r%s %.2fs/step Step %d: %s." %
                             (self.params['phase'], times_mean, batch, metrics_str))
            sys.stdout.flush()
            self.last_step = batch


