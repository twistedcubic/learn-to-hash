
import os, sys, math
import numpy as np
#from sklearn.cluster import MiniBatchKMeans, KMeans
import utils
import pickle
import time
#import json
#from collections import defaultdict
#import kahip
import os.path as osp

import pdb

data_dir = 'data'
#n_clusters = 64
km_method = 'km' #mbkm km
max_loyd = 10

'''
Implements solver fit and transform functionality.
'''
class KahipSolver():

        def __init__(self):
                #self.n_clusters = n_clusters
                #kahip partition top level result
                #64 for now!!
                #-loads partition data and prepares to make predictions
                self.kahip_path = osp.join(utils.data_dir, 'cache_partition64strong_0ht2')
                classes_l = utils.load_lines(self.kahip_path) ##########
                self.classes_l = [int(c) for c in classes_l]                
                
        '''
        predict data
        Input:
        -dataset_idx: indices of data. List of ints.
        Output:
        - classes for index, as numpy array
        '''
        def predict(self, dataset_idx):
                #needs to predict classes: d_cls_idx = solver.predict(dataset)
                #should use numpy array for efficiency!
                pred_ar = np.zeros(len(dataset_idx))
                for i, idx in enumerate(dataset_idx):
                        pred_ar[i] = self.classes_l[idx]
                return pred_ar
                

if __name__ == '__main__':
        opt = utils.parse_args()
        n_clusters = opt.n_clusters
        n_clusters = 2
        KahipSolver(n_clusters, opt)
