'''
Subset of driver functions used for evaluating models. Such as tuning nn_mult.
'''
import _init_paths
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.data import Dataset
import main, utils
import torch.optim as optim
import sys
import numpy as np
import os.path as osp
from datetime import date
import train

import pdb

device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
if __name__ == '__main__':
    
    opt = utils.parse_args()

    if True:
        if opt.glove:
            queryset = utils.load_glove_data('query').to(utils.device)
            neighbors = utils.load_glove_data('answers').to(utils.device)
        elif opt.sift:
            queryset = utils.load_sift_data('query').to(utils.device)
            neighbors = utils.load_sift_data('answers').to(utils.device)    
        else:
            queryset = utils.load_data('query').to(utils.device)
            neighbors = utils.load_data('answers').to(utils.device)
    else:
        queryset = utils.load_data('train').to(utils.device)        
        dist = utils.l2_dist(queryset)
        dist += 2*torch.max(dist).item()*torch.eye(len(dist)) #torch.diag(torch.max(dist))
        val, neighbors = torch.topk(dist, k=opt.k, dim=1, largest=False)
            
    if False:
        trainset = utils.load_data('train').to(utils.device)       
        dist = utils.l2_dist(queryset, trainset)
        #dist += 2*torch.max(dist).item()*torch.eye(len(dist)) #torch.diag(torch.max(dist))
        val, neighbors = torch.topk(dist, k=opt.k, dim=1, largest=False)
        
    height = 1
    n_bins_l = list(range(1, 45, 2))
    n_bins_l = list(range(1, 100))
    n_bins_l = list(range(1, 35, 1)) #[1]
    n_clusters_l = [16]#[16] #[2]
        
    acc_mx = np.zeros((len(n_clusters_l), len(n_bins_l)))
    probe_mx = np.zeros((len(n_clusters_l), len(n_bins_l)))
    probe95_mx = np.zeros((len(n_clusters_l), len(n_bins_l)))
    col_max = 0
    mult_l = list(range(2, 12))
    mult_l = [1,3,4,5,6,7,8,9,10,11,12]
    
    mult_l = [.1, .5, .9, 1.1, 1.5 ]
    mult_l = [2]
    for cur_mult in mult_l:
        opt.nn_mult = cur_mult        
        for i, n_clusters in enumerate(n_clusters_l):
            for j, n_bins in enumerate(n_bins_l):
                acc, probe_count, probe_count95 = train.deserialize_eval(queryset, neighbors, height, n_clusters, n_bins, opt)
                
                acc_mx[i][j] = acc
                probe_mx[i][j] = probe_count
                probe95_mx[i][j] = probe_count95
                if acc > 0.95:
                    break
            if j > col_max:
                col_max = j

        acc_mx = acc_mx[:, :col_max+1]
        probe_mx = probe_mx[:, :col_max+1]
        probe95_mx = probe95_mx[:, :col_max+1]

        row_label = ['{} clusters'.format(i) for i in n_clusters_l[:col_max+1]]
        col_label = ['{} bins'.format(i) for i in n_bins_l[:col_max+1]]
        acc_md = utils.mxs2md([np.around(acc_mx,3), np.rint(probe_mx), np.rint(probe95_mx)], row_label, col_label)

        if opt.write_res:
            if opt.glove:
                res_path = osp.join('results', 'glove_train_S.md')
            elif opt.sift:
                res_path = osp.join('results', 'sift_train_S.md')
            else:
                res_path = osp.join('results', 'mnist_train_S.md')
            with open(res_path, 'a') as file:
                file.write('\n\n{} **Training. MLCE. {} neighbors, k_graph: {}, k: {}, height: {}, nn_mult: {}** \n\n'.format(str(date.today()), opt.nn_mult*opt.k, opt.k_graph, opt.k, height, opt.nn_mult))
                file.write(acc_md)
                                                        
    
    

