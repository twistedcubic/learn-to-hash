
'''
Utility functions to create knn graphs from data, as well as knn subgraphs from an existing graph..
'''
import _init_paths
import torch
import numpy as np
import os
import pdb
from collections import defaultdict, Counter
import utils
import os.path as osp

import pdb

DEBUG = False

data_dir = utils.data_dir
graph_file = 'knn.graph'

'''
Normalize input data and create graph
Input:
-data: tensor of coordinates/features
Returns:
-ranks, tensor 
'''
def create_knn_graph(data, k, opt=None):

    if opt != None and hasattr(opt, 'ranks_path'):
        ranks = np.load(opt.ranks_path)
        ranks = torch.from_numpy(ranks)
        pdb.set_trace()
    elif opt != None and opt.normalize_data:
        '''
        data /= data.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-3)
        dist = torch.matmul(data, data.t())# - torch.eye(len(data))
        val, ranks = torch.topk(dist, k=k+1, dim=1)
        ranks = ranks[:, 1:]
        '''
        ranks = utils.dist_rank(data, k=k, opt=opt)
    else:
        #compute l2 dist <--be memory efficient by blocking
        '''
        dist = utils.l2_dist(data)        
        dist += 2*torch.max(dist).item()*torch.eye(len(data))
        val, ranks = torch.topk(dist, k=k, dim=1, largest=False)
        '''
        ranks = utils.dist_rank(data, k=k, opt=opt)
           
        if DEBUG:
            print(dist)
                    
    #add 1 since the indices for kahip must be 1-based.
    ranks += 1
    
    return ranks

'''
Note: ds_idx is 0-based 1D tensor, but ranks is 1-based 2D tensor
ds_idx are indices of current data in all of dataset. 0-based.
idx2weights are 1-based indices.

local_ranks: Closest point from every point in ds_idx to some other point in ds_idx
local_ranks also 1-based.
all_ranks: can be tensor or python list

'''
def create_knn_sub_graph(all_ranks, idx2weights, ds_idx, data, opt):

    if False:
        if opt != None and opt.normalize_data and not opt.glove:
            data /= data.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-3)
            dist = torch.matmul(data, data.t()) - torch.eye(len(data))
            val, local_ranks = torch.topk(dist, k=1, dim=1)
        else:
            '''
            dist = utils.l2_dist(data)        
            dist = dist + 2*torch.max(dist).item()*torch.eye(len(data))
            val, local_ranks = torch.topk(dist, k=1, dim=1, largest=False)
            '''
            #just compute this in the loop since not too frequent
            local_ranks = utils.dist_rank(data, k=1, opt=opt)
        
        local_ranks += 1
    
    ranks = []
    #is_tensor = isinstance(all_ranks, torch.Tensor) 
    ds_idx2idx = {t.item()+1 : idx for idx, t in enumerate(ds_idx, 1)}
    #dict of idx of a point to its nearest neighbor
    idx2nn = {}
    #set of added tuples
    added_tups = set()

    if isinstance(all_ranks, torch.Tensor) :
        all_ranks = all_ranks.cpu().numpy()
    if isinstance(ds_idx, torch.Tensor):
        ds_idx = ds_idx.cpu().numpy()
    for idx, i in enumerate(ds_idx):
        cur_ranks = []
        for j in all_ranks[i]:
            
            tup = (i+1, j) if i+1 < j else (j, i+1)
            if idx2weights[tup] == 1 and tup in added_tups:
                continue
            
            added_tups.add(tup)
            
            if j in ds_idx2idx:
                cur_ranks.append(ds_idx2idx[j])
        
        if cur_ranks == []:
            #nearest_idx = local_ranks[idx][0].item()

            local_ranks = utils.dist_rank(data[idx].unsqueeze(0), k=1, data_y=data, opt=opt)
            nearest_idx = local_ranks[0][0].item() + 1
            cur_ranks.append(nearest_idx)
            idx2nn[idx] = nearest_idx
        
        ranks.append(cur_ranks)
    for idx, nn in idx2nn.items():
        ranks[nn-1].append(idx+1)
        
   
    return ranks
    
'''
Save knn graph in format used by kahip.
Input: 
-ranks: tensor or list. Content indices are 1-based. 
#kahip takes lines corresponding to points, each line contains indices of neighbors
'''
def write_knn_graph(ranks, path):
    #torch.save(ranks, path+'.pth')
    #edge_set = set()
    edge_l = []
    #edge_l = [[([i, j] if i < j else [j, i]) for j in row] for i,row in enumerate(ranks)]
    #print(ranks.size())
    
    is_tensor = isinstance(ranks, torch.Tensor)
    print('in write knn graph')
    pdb.set_trace()
    
    
    if is_tensor:
        ##[[(edge_l.append((i, j.item())) if i < j.item() else edge_l.append((j.item(), i))) for j in row] for i,row in enumerate(ranks,1)]
        weighted_n_edges = ranks.size(0) * ranks.size(1)
        ranks = ranks.cpu().numpy()
        [[(edge_l.append((i, j)) if i < j else edge_l.append((j, i))) for j in row] for i,row in enumerate(ranks,1)]
    else:
        [[(edge_l.append((i, j)) if i < j else edge_l.append((j, i))) for j in row] for i,row in enumerate(ranks,1)]
        weighted_n_edges = np.sum(np.sum([len(row) for row in ranks]))

    print('weighted_n_edges ', weighted_n_edges)
    edge_counter = Counter(edge_l)
    n_edges = len(edge_counter)
    
    if DEBUG:
        print('rank ', ranks)
        print('edge counter {}'.format(edge_counter))
    print('edge_set len {}'.format(n_edges))
    #use os.line_sep 
    ##s = '\n'.join([' '.join([str(i.item()) for i in row]) for row in ranks])
    d = defaultdict(set)
    
    for i, row in enumerate(ranks, 1):
        for j in row:
            if j == i:
                continue
            d[i].add(j)
            d[j].add(i)
    #ensure graph symmetry for kahip
    s2 = '\n'.join([' '.join([str(i) + (' 2' if max(edge_counter[(row,i)], edge_counter[(i,row)])==2 else ' 1') for i in d[row]]) for row in range(1, len(ranks)+1)])
    
    #[' '.join([str(i) + (' 2' if max(edge_counter[(row,i)], edge_counter[(i,row)])==2 else ' 1') for i in d[row]]) for row in range(1, len(ranks)+1)]
    
    #print('edge_counter ', edge_counter )
    #1 indicates the graph only uses edge weights
    s2 = str(len(ranks)) + ' ' + str(n_edges) + ' 1 \n' + s2
    with open(path, 'w') as file:
        file.write(s2)
        print('written to ', path)
    return weighted_n_edges

def deserialize_create_graph():
    opt = utils.parse_args()
    
    ##data = torch.from_numpy(np.load('../data/queries_unnorm.npy'))
    #data = torch.from_numpy(np.load(osp.join(utils.data_dir, 'sift_dataset_unnorm.npy')))
    dataset_name = 'prefix10m' #'glove'
    #opt.ranks_path = 'data/{}_answers.npy'.format(dataset_name)
    
    #10 is subsampling frequency
    data = torch.from_numpy(np.load(osp.join(utils.data_dir, '{}_dataset.npy'.format(dataset_name))))
    #subsample_ = False
    subsample = 0 #10
    if subsample > 0:        
        sub_idx = torch.randperm(len(data))[:int(len(data)/subsample)]
        torch.save(sub_idx, 'data/sub10_glove_idx.pt')
        data = data[sub_idx]
        pdb.set_trace()
    #data = data[192424:192436]
    #data[-1] = data[1]
    k=10
    if subsample > 0:
        path_to = os.path.join(utils.data_dir, '{}{}_sub{}'.format(dataset_name, k, subsample)+graph_file) #'../data/knn.graph'
    else:
        path_to = os.path.join(utils.data_dir, '{}{}'.format(dataset_name, k)+graph_file) #'../data/knn.graph'
    
    ranks = create_knn_graph(data, k, opt)
    ##torch.save(ranks, '/large/ranks.pt')
    print('done creating graph. path to graph {}'.format(path_to))
    if True:
        s = write_knn_graph(ranks, path_to)

if __name__=='__main__':
    
    if True:
        deserialize_create_graph()
    else:
        data = torch.FloatTensor([[1],[2],[3],[4],[5],[6],[7]])
        #data = torch.FloatTensor([[1],[2],[3]])
        k=1
        ranks = create_knn_graph(data, k)
        s = write_knn_graph(ranks, 'test_knn_graph')

