
from sklearn.decomposition import PCA
import torch
import numpy as np
import utils
import torch.sparse

import pdb

'''
Classes for two linear models.
Linear model: PCA, used to compared with trained supervised models learned from kahip partitions.
Linear model: random projections, used to compared with trained supervised models learned using kahip partitions.
'''

class PCASolver():

    #Guaranteed to only need top component
    def __init__(self, dataset, opt):

        if isinstance(dataset, torch.Tensor):
            dataset = torch.tensor(dataset).cpu().numpy()
        
        #scale features        
        dataset_t = np.transpose(dataset)
        self.ds_mean = dataset_t.mean(axis=-1, keepdims=True)
        dataset_t = dataset_t - self.ds_mean
        self.ds_std = dataset_t.std(axis=-1, keepdims=True).clip(min=0.1)
        dataset_t = dataset_t / self.ds_std
        dataset = np.transpose(dataset_t)
        
        self.pca = PCA(n_components=1)
        #shape: n_sample x 1
        self.pca.fit(dataset)
        out = self.pca.transform(dataset)
        
        self.median = np.median(out)

    '''
    Input: k here to satisfy uniform interface with kmeans solver.
    query: 2D vec
    Output:
    -1 D np array
    '''
    def predict(self, query):
        
        if isinstance(query, torch.Tensor):
            query = torch.tensor(query).cpu().numpy()
            
        cls = np.zeros(len(query))
        query = np.transpose((np.transpose(query) - self.ds_mean) / self.ds_std)
        out = self.pca.transform(query).reshape(-1)

        cls[out >= self.median] = 1
        return cls

from scipy.stats import ortho_group
'''
Another linear method, random projection.
'''
class RPSolver():

    def __init__(self, dataset, opt):
        
        if isinstance(dataset, torch.Tensor):
            dataset = torch.tensor(dataset).cpu().numpy()

        self.data_mean = dataset.mean(axis=0)
        #orthogonal projection
        self.orth_mx = ortho_group.rvs(dataset.shape[-1])
        #self.rand_vec = np.random.randn(dataset.shape[-1]) #np.random.multivariate_normal(mean, cov)
        
    '''
    Input: k here to satisfy uniform interface with kmeans solver.
    query: 2D vec
    Output:
    -1 D np array
    '''
    def predict(self, query):
        
        if isinstance(query, torch.Tensor):
            query = torch.tensor(query).cpu().numpy()
        
        query = query - self.data_mean
                
        out = np.matmul(query, self.orth_mx).sum(axis=-1)
        cls = np.zeros(len(query))
        cls[out > 0] = 1
        return cls

'''
Search tree solver.
'''
class STSolver():
    
    '''
    Input:
    -dataset: dataset for current node, ie subset of full dataset.
    -knn_graph: knn graph
    -ranks: nearest neighbor ranks matrix (as original distances), indices are as original dataset. ranks
    for index i includes the i itself.
    -idx: indices of dataset used in cur iteration, indices are wrt original dataset.
    '''
    def __init__(self, dataset, ranks, idx, opt):

        if isinstance(dataset, np.ndarray):
            dataset = torch.from_numpy(dataset).to(utils.device)
            idx = torch.from_numpy(idx).to(utils.device)
        #augment last component with 1's
        dataset = torch.cat((dataset, torch.zeros(len(dataset), 1, device=utils.device)), dim=-1)

        if len(dataset) != len(ranks):
            long_vec = -torch.ones(len(ranks), device=utils.device)        
            src_vec = torch.cuda.FloatTensor(range(len(idx)))      
            long_vec.scatter_(dim=0, index=idx, src=src_vec)
            long_vec = long_vec.cpu().numpy()

            #sparse_idx = torch.LongTensor(len(idx )  )
            sparse_idx_l = []
            for i, t in enumerate(idx):
                cur_vec = []
                for j in ranks[t]:
                    if long_vec[j] != -1:
                        cur_vec.append(long_vec[j])

                idx_i = torch.cat((torch.ones(1, len(cur_vec), dtype=torch.int64, device=utils.device)*i, torch.cuda.LongTensor(cur_vec).unsqueeze(0)), dim=0)
                sparse_idx_l.append(idx_i)

            #2 x number of non-zero entries
            sparse_idx = torch.cat(sparse_idx_l, dim=-1)
        else:
            #pdb.set_trace()
            range_vec = torch.cuda.LongTensor(range(len(dataset))).unsqueeze(-1).repeat(1, ranks.size(-1))
            ranks = ranks.to(utils.device)
            sparse_idx = torch.cat((range_vec.view(1, -1), ranks.view(1, -1)), dim=0)
            
        
        sparse_idx1 = torch.clone(sparse_idx)
        sparse_idx1[0] = sparse_idx[1]
        sparse_idx1[1] = sparse_idx[0]
        
        sparse_idx = torch.cat((sparse_idx, sparse_idx1), dim=-1)        
        sparse_val = torch.ones(sparse_idx.size(-1), device=utils.device)

        sparse_vec = torch.sparse.FloatTensor(sparse_idx, sparse_val, torch.Size([len(dataset), len(dataset)]))
        sparse_vec = sparse_vec.coalesce()
        
        sparse_vec = torch.sparse.FloatTensor(sparse_vec._indices(), torch.ones_like(sparse_vec._values()), torch.Size([len(dataset), len(dataset)]) )        
        
        lamb = sparse_vec._values().sum().item()/len(dataset)**2  #.001
        print('lamb {}'.format(lamb))
        ones = torch.ones(1, dataset.size(0), device=utils.device)
        W = torch.mm(torch.sparse.mm(sparse_vec.t(), dataset).t(), dataset) - lamb*torch.mm(torch.mm(dataset.t(), ones.t()), torch.mm(ones, dataset))
        eval_, evec_ = torch.eig(W, eigenvectors=True)
        eval_ = eval_[:, 0]
        evec_ = evec_.t()
        #pdb.set_trace()
        max_idx = torch.argmax(eval_)
        self.top_evec = evec_[max_idx]
        self.top_evec = self.top_evec.cpu().numpy()
        
        
    '''
    Input: k here to satisfy uniform interface with kmeans solver.
    query: 2D vec
    Output:
    -1 D np array
    '''
    def predict(self, query):

        query = np.concatenate((query, np.ones((len(query), 1))), axis=-1)
        projected = (self.top_evec * query).sum(-1)
        
        cls = np.zeros(len(query))
        cls[projected > 0] = 1
        #print('sum! {}'.format( cls.sum()))
        #pdb.set_trace()
        return cls

