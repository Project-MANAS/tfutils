from functools import partial
from math import inf

import tensorflow as tf

class ReduceLROnPleateau:
    def __init__(self, starting_lr=1e-3, mode='min', factor=0.1, 
            patience=10, verbose=False, threshold=0.05, 
            threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8):
        
        self.old_lr = starting_lr
        self.factor = factor
        self.min_lr = min_lr
        self.patience = patience
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None
        self.verbose = verbose
        self.is_better = None
        self.eps = eps
        self.last_epoch = -1
        self._init_is_better(mode, threshold, threshold_mode)
        self._reset()

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode == 'min':
            self.mode_worse = inf
        else:
            self.mode_worse = -inf

        self.is_better = partial(self._cmp, mode, threshold, threshold_mode)

    def _cmp(self, mode, threshold, threshold_mode, a, best):
        if mode == 'min' and threshold_mode == 'rel':
            rel_eps = 1. - threshold
            return a < best * rel_eps

        elif mode == 'min' and threshold_mode == 'abs':
            return a < best - threshold

        elif mode == 'max' and threshold_mode == 'rel':
            rel_eps = 1. + threshold
            return a > best * rel_eps

        else:
            return a > best + threshold

    def _reset(self):
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epoch = 0

    def __call__(self, metrics):
        current = metrics
        epoch = self.last_epoch = self.last_epoch + 1

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0

        else:
            self.num_bad_epochs += 1

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        return self.old_lr

    def _reduce_lr(self, epoch):
        new_lr = max(self.old_lr * self.factor, self.min_lr)
        if self.old_lr - new_lr > self.eps:
            if self.verbose:
                print('Reducing learning rate on epoch', epoch, 'from', self.old_lr, 'to', new_lr)
            self.old_lr = new_lr
