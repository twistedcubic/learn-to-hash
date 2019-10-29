'''
Used for tuning various hyperparameter, such as nn_mult, the multiplicity used for 
sampling neighbors during soft label creation.
'''

import _init_paths
import sys
import os
import os.path as osp
import pickle
import create_graph
import torch
import numpy as np
import argparse
import utils
import math
import kmkahip
from model import train
from data import DataNode
import utils
from collections import defaultdict
import multiprocessing as mp
import kmeans


if __name__ == '__main__':
    opt = utils.parse_args()

    opt.n_clusters = 256 #256
    opt.n_class = opt.n_clusters
    print('number of bins {}'.format(opt.n_class))
    
    mult_l = [1,2,3,4,5,6,7,8,9,10,11,12]
    #mult_l = [1,3,5,7,9,11]
    #mult_l = [4,6,8,10,12]
    mult_l = [.1, .5, .9, 1.1, 1.5]
    # This is now set upstream, keep here for demo purposes.
    # actions can be km, kahip, train, logreg #
    opt.level2action = {0:'km', 1:'train'} 
    opt.level2action = {0:'train', 1:'train'}         
    
    opt.level2action = {0:'logreg', 2:'logreg', 3:'logreg', 4:'logreg', 5:'logreg', 6:'logreg', 7:'logreg', 8:'logreg', 9:'logreg', 10:'logreg', 11:'logreg'}
    opt.level2action = {0:'train', 1:'train'}
    
    height_l = range(1, 9)
    height_l = [1]
    if opt.glove:
        dataset = utils.load_glove_data('train').to(utils.device)
        queryset = utils.load_glove_data('query').to(utils.device)    
        neighbors = utils.load_glove_data('answers').to(utils.device)
        opt.dataset_name = 'glove'
    elif opt.glove_c:
        #catalyzer glove vecs
        dataset = utils.load_glove_c_data('train').to(utils.device)
        queryset = utils.load_glove_data('query').to(utils.device)    
        neighbors = utils.load_glove_data('answers').to(utils.device)
        opt.dataset_name = 'glove'
        opt.glove = True
    elif opt.sift:
        dataset = utils.load_sift_data('train').to(utils.device)
        queryset = utils.load_sift_data('query').to(utils.device)    
        neighbors = utils.load_sift_data('answers').to(utils.device)
        opt.dataset_name = 'sift'
    else:
        dataset = utils.load_data('train').to(utils.device)
        queryset = utils.load_data('query').to(utils.device)    
        neighbors = utils.load_data('answers').to(utils.device)
        opt.dataset_name = 'mnist'
        
    for mult in mult_l:
        print('cur nn_mult: {}'.format(mult))
        opt.nn_mult = mult
        for height in height_l:
            kmkahip.run_kmkahip(height, opt, dataset, queryset, neighbors)
