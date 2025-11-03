'''
This file contains the implementation of Stochastic Weight Averaging (SWA) for model parameters.
'''
import torch
import torch.nn as nn

class SWA:
    '''
    Stochastic Weight Averaging for model parameters.
    '''
    def __init__(self, model):
        self.model = model
        self.shadow = {}
        self.n_models = 0
        self.backup = {}

    def register(self):
        '''
        Register model parameters for SWA.
        '''
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        self.n_models = 1

    def update(self):
        '''
        Update SWA parameters.
        '''
        self.n_models += 1
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (self.shadow[name] * (self.n_models - 1) + param.data) / self.n_models
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        '''
        Apply shadow parameters to the model.
        '''
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        '''
        Restore model parameters from backup.
        '''
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
