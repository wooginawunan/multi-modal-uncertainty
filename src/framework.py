import warnings
import numpy as np
import timeit
import torch
import math
import itertools

from src.callbacks import (
     ValidationProgressionCallback, 
     ProgressionCallback,
     CallbackList, 
)

import logging
logger = logging.getLogger(__name__)

warning_settings = {
    'batch_size': 'warn'
}

def cycle(iterable): 
    while True:
        for x in iterable:
            yield x


def _get_step_iterator(steps, generator):
    count_iterator = range(1, steps + 1) if steps is not None else itertools.count(1)
    generator = cycle(generator) if steps is not None else generator
    return zip(count_iterator, generator)


class StepIterator:
    def __init__(self, generator, steps_per_epoch, callback, metrics_names):
        self.generator = generator
        self.steps_per_epoch = steps_per_epoch
        self.callback = callback
        self.metrics_names = metrics_names

        self.losses_sum = 0.
        self.metrics_sum = np.zeros(len(self.metrics_names))
        self.sizes_sum = 0.
        self.extra_lists = {}

        self.defaultfields = [
                              'loss', 
                              'metrics', 
                              'number',
                              'size'
                              ]

    @property
    def loss(self):
        return self.losses_sum / self.sizes_sum if self.sizes_sum!=0 else 0

    @property
    def metrics(self):
        
        if self.sizes_sum==0:
            return dict(zip(self.metrics_names, np.zeros(len(self.metrics_names))))
        else:
            metrics_dict = dict(zip(self.metrics_names, self.metrics_sum / self.sizes_sum))
            return metrics_dict

    def __iter__(self):
        for batch_ind, data in _get_step_iterator(self.steps_per_epoch, self.generator):
            batch_begin_time = timeit.default_timer()
            self.callback.on_batch_begin(batch_ind, {})
            self.callback.on_forward_begin(batch_ind, data) 

            step_data = {'number': batch_ind}
            yield step_data, data

            self.losses_sum += step_data['loss'] * step_data['size']
            self.metrics_sum += step_data['metrics'] * step_data['size']
            self.sizes_sum += step_data['size']

            metrics_dict = dict(zip(self.metrics_names, step_data['metrics']))
            
            for key, value in step_data.items():
                if key not in self.defaultfields:
                    if key in self.extra_lists:
                        self.extra_lists[key].append(value)
                    else:
                        self.extra_lists[key] = [value]
                    
            batch_total_time = timeit.default_timer() - batch_begin_time

            batch_logs = {'batch': batch_ind, 'size': step_data['size'], 
                          'time': batch_total_time, 'batch_begin_time': batch_begin_time, 
                          'loss': step_data['loss'], **metrics_dict}

            self.callback.on_batch_end(batch_ind, batch_logs)


class Model_:
    def __init__(self, model, optimizer, scheduler, data_forming_func, *, metrics=[], 
        verbose=True):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.data_forming = data_forming_func
        self.metrics = metrics
        self.metrics_names = [metric.__name__ for metric in self.metrics]
        self.device = None
        self.verbose = verbose
        self.verbose_logs = {} 

    def _compute_metrics(self, pred_y, y, eval):
        return np.array([float(metric(pred_y, y, eval)) for metric in self.metrics])

    def _transfer_optimizer_state_to_right_device(self):
        # Since the optimizer state is loaded on CPU, it will crashed when the
        # optimizer will receive gradient for parameters not on CPU. Thus, for
        # each parameter, we transfer its state in the optimizer on the same
        # device as the parameter itself just before starting the optimization.
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p in self.optimizer.state:
                    for _, v in self.optimizer.state[p].items():
                        if torch.is_tensor(v) and p.device != v.device:
                            v.data = v.data.to(p.device)

    def to(self, device):
        self.device = device
        self.model.to(self.device)

        for metric in self.metrics:
            if isinstance(metric, torch.nn.Module):
                metric.to(self.device)

        return self

    def eval_loop(self, generator, phase, *, steps=None):
        if steps is None:
            steps = len(generator)
        
        step_iterator = StepIterator(
                            generator, 
                            steps, 
                            ValidationProgressionCallback(
                                phase=phase, 
                                steps=steps, 
                                metrics_names=['loss'] + self.metrics_names
                            ), 
                            self.metrics_names,
                        )

        self.model.eval()
        with torch.no_grad():
            for step, (x, y) in step_iterator:
                step['size'] = len(x)
                x, y = self.data_forming(x, y, phase='eval')
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                loss = self.model.compute_loss(outputs, y, eval=True)
                info = self._compute_metrics(outputs, y, eval=True)
                        
                step['loss'] = float(loss)
                step.update({'metrics': info})

        metrics_dict = {
            f'{phase}_{metric_name}' : metric for metric_name, metric in step_iterator.metrics.items()
        }

        info_dict = {f'{phase}_loss' : step_iterator.loss, 
            **{f'{phase}_{k}':v for k, v in step_iterator.extra_lists.items()},
            **metrics_dict
        }

        return info_dict

    def train_loop(self,
                      train_generator,
                      test_generator=None,
                      valid_generator=None,
                      *,
                      epochs=1000,
                      steps_per_epoch=None,
                      validation_steps=None,
                      test_steps=None,
                      patience=10, # early stopping
                      callbacks=[],
                      ):
        
        self._transfer_optimizer_state_to_right_device()

        callback_list = CallbackList(callbacks)
        callback_list.append(ProgressionCallback())
        callback_list.set_params({'epochs': epochs, 'steps': steps_per_epoch})
        callback_list.set_model_pytoune(self)
        
        stop_training = False
        
        stopped_epoch, counter = 0, 0
    
        callback_list.on_train_begin({})
        val_dict, test_dict = {}, {}
        for epoch in range(1, epochs+1):
            callback_list.on_epoch_begin(epoch, {})
            
            epoch_begin_time = timeit.default_timer()
            # training
            train_step_iterator = StepIterator(train_generator,
                                               steps_per_epoch,
                                               callback_list,
                                               self.metrics_names,
                                               )
            self.model.train(True)
            with torch.enable_grad():
                for step, (x, y) in train_step_iterator: 
                    
                    x, y = self.data_forming(x, y, phase='train')
                    step['size'] = len(x)
                    self.optimizer.zero_grad()

                    x, y = x.to(self.device), y.to(self.device)
                    y_pred  = self.model(x) # B, E, C
                    loss = self.model.compute_loss(y_pred, y)
                    loss.backward()

                    with torch.no_grad():
                        info = self._compute_metrics(y_pred, y, eval=False)
                    callback_list.on_backward_end(step['number'])
                    self.optimizer.step()  
                    step.update({'metrics': info})
                    step['loss'] = loss.item()
                    
                    if math.isnan(step['loss']): stop_training = True

            train_dict = {'loss': train_step_iterator.loss, 
                    **{f'train_{k}':v for k, v in train_step_iterator.extra_lists.items()},
                    **train_step_iterator.metrics}
            
            # validation
            val_dict = self.eval_loop(valid_generator, 'val', steps=validation_steps)
            # test
            test_dict = self.eval_loop(test_generator, 'test', steps=test_steps)
           
            epoch_log = {
                'epoch': epoch, 
                'time': timeit.default_timer() - epoch_begin_time, 
                'epoch_begin_time': epoch_begin_time,
                **train_dict, **val_dict, **test_dict
            }
            
            self.scheduler.step(epoch_log['loss'])
             
            callback_list.on_epoch_end(epoch, epoch_log)
            
            if epoch_log['acc'] == 100: counter +=1
                
            if counter>=patience:
                stopped_epoch, stop_training = epoch, True
                
            if stop_training: break
            
        callback_list.on_train_end({})
        
        if stopped_epoch > 0:
            print('Epoch %05d: completed stopping' % (stopped_epoch))

