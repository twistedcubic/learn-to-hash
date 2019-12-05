
from sklearn.svm import LinearSVC
import torch
import numpy as np
import sklearn.linear_model

import pdb

'''
Implements logistic regression solver for uniform solver interface.
'''

class LogReg():
    def __init__(self, dataset, labels, opt):
        
        if isinstance(dataset, torch.Tensor):
            dataset = torch.tensor(dataset).cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels).cpu().numpy()

        self.normalize_feat = False        
        if self.normalize_feat:
            dataset_t = np.transpose(dataset)
            self.ds_mean = dataset_t.mean(axis=-1, keepdims=True)
            dataset_t = dataset_t - self.ds_mean
            self.ds_std = dataset_t.std(axis=-1, keepdims=True).clip(min=0.1)
            dataset_t = dataset_t / self.ds_std
            dataset = np.transpose(dataset_t)        

        self.svc = sklearn.linear_model.LogisticRegression()
        self.svc.fit(dataset, labels)

    '''
    Input: 
    query: 2D data matrix.
    k: here to satisfy uniform interface with solver.
    '''
    def predict(self, query, k=None):
        
        if isinstance(query, torch.Tensor):
            query = torch.tensor(query).cpu().numpy()

        if self.normalize_feat:
            query = np.transpose((np.transpose(query) - self.ds_mean) / self.ds_std)
        return self.svc.predict(query)
