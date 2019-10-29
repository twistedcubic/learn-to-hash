import _init_paths
import numpy as np
import torch
import random
import sys
import utils
import _multiprobe

import pdb

'''
Class for cross polytope LSH.
'''
device = utils.device 
device_cpu = torch.device('cpu')

class CPLshSolver:

    '''
    n_clusters should be power of 2 for cross polytope LSH
    '''
    def __init__(self, dataset, n_clusters, opt):
                
        if isinstance(dataset, np.ndarray):
            dataset = torch.from_numpy(dataset).to(utils.device)

        n_poly = 0
        n_c = n_clusters
        while n_c > 1:
            n_c /= 2
            n_poly += 1
        #pdb.set_trace()
        if n_c != 1:
            pdb.set_trace()
        
        assert n_c == 1
        #second parameter is number of bits, last parameter is seed.
        self.cplsh = _multiprobe.Multiprobe(dataset.size(-1), n_poly, 4057218)
        
        #self.centers, self.codes = self.build_kmeans(dataset, n_clusters)
        #self.centers_norm = torch.sum(self.centers**2, dim=0).view(1,-1).to(utils.device)
        self.opt = opt
        
        
    '''
    Input: query. tensor, batched query.
    Returns:
    -indices of nearest centers
    '''
    def predict(self, query, k):

        res = -np.ones((len(query), k))
        
        for i, q in enumerate(query):
            #res[i] = self.cplsh.query(q)[:k]
            cur_res = self.cplsh.query(q)            
            cur_res_len = min(k, len(cur_res))
            res[i][:cur_res_len] = cur_res[:cur_res_len]            

        return res
    
    '''
        #if isinstance(query, np.ndarray):
        #    query = torch.from_numpy(query).to(utils.device)
        
        #self centers have dimension 1, torch.Size([100, 1024])
        if hasattr(self, 'opt') and (self.opt.glove or self.opt.sift) and self.centers.size(1) > 512:
            
            centers = self.centers.t()
            idx = utils.dist_rank(query, k, data_y=centers, largest=False)
        else:
            
            q_norm = torch.sum(query ** 2, dim=1).view(-1, 1)
            dist = q_norm + self.centers_norm - 2*torch.mm(query, self.centers)
        
            if k > dist.size(1):
                k = dist.size(1)
            _, idx = torch.topk(dist, k=k, dim=1, largest=False)
            #move predict to numpy
        
        idx = idx.cpu().numpy()
        return idx    
    '''

if __name__ == '__main__':
    
    dataset_numpy = np.load('dataset.npy')
    queries_numpy = np.load('queries.npy')
    answers_numpy = np.load('answers.npy')
    
    dataset = torch.from_numpy(dataset_numpy).to(device)
    queries = torch.from_numpy(queries_numpy).to(device)
    answers = torch.from_numpy(answers_numpy)
    
