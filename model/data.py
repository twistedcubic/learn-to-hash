
'''
Create data tree from input data, where each node contains train dataloader
To be used for training.
'''
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

import pdb

#convert to parse args eventually
batch_size = 64
shuffle = True
train_split = 0.99 #set to 99% now that done with neural net tuning

class DataNode():
    '''
    Use kahit output to create train loader
    When writing to file, write data in .
    Input:
    -ds_idx are indices amongst entire dataset.
    -kahip_path: path to kahip output 
    '''
    def __init__(self, ds_idx, y, n_class, ranks=None):#, n_input, n_hidden , n_class):
        #self.idx = idx
        
        try:
            assert len(ds_idx) == len(y)
        except AssertionError:
            print('len(ds_idx) != len(y)')
            pdb.set_trace()
            raise AssertionError('len(ds_idx) != len(y)')
            
        datalen = len(y)
        cur_split = int(datalen*train_split)
        
        #y are cluster labels.
        if ranks is not None:
                    
            device = y.device
            ranks = ranks.to(device)
            y_exp = y.unsqueeze(0).expand(datalen, -1)
            #datalen x opt.k (or the number of nearest neighbors to take for computing acc)
            neigh_cls = torch.gather(y_exp, 1, ranks)
            neigh_cls = torch.cat((neigh_cls, y.unsqueeze(-1)), dim=1)

            cls_ones = torch.ones(datalen, neigh_cls.size(-1), device=device)
            cls_distr = torch.zeros(datalen, n_class, device=device)
            #datalen x opt.k
            cls_distr.scatter_add_(1, neigh_cls, cls_ones)
            cls_distr /= neigh_cls.size(-1)

            trainset = TensorDataset(ds_idx[:cur_split], y[:cur_split], cls_distr[:cur_split])
        else:
            trainset = TensorDataset(ds_idx[:cur_split], y[:cur_split])
            
        self.trainloader = DataLoader(dataset=trainset, batch_size=batch_size,
                                       shuffle=True)

        #validation set
        valset = TensorDataset(ds_idx[cur_split:], y[cur_split:])
        self.valloader = DataLoader(dataset=valset, batch_size=64, shuffle=False)
        
        self.children = []
        self.n_class = n_class
        
    def add_child(self, node):
        
        self.children.append(node)
