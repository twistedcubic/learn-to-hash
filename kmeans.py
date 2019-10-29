import numpy as np
import torch
import random
import sys
import utils
import pdb

'''
Class for running Lloyd iterations for k-means and associated utility functions.
'''

chunk_size = 8192
num_iterations = 60
k = 10
device = utils.device 
device_cpu = torch.device('cpu')

class FastKMeans:

    def __init__(self, dataset, n_clusters, opt):
        if isinstance(dataset, np.ndarray):
            dataset = torch.from_numpy(dataset).to(utils.device)
        self.centers, self.codes = self.build_kmeans(dataset, n_clusters)
        self.centers_norm = torch.sum(self.centers**2, dim=0).view(1,-1).to(utils.device)
        self.opt = opt
        #self.k = opt.k
        
    '''
    Creates kmeans
    '''
    def build_kmeans(self, dataset, num_centers):
        return build_kmeans(dataset, num_centers)
        
    '''
    Input: query. tensor, batched query.
    Returns:
    -indices of nearest centers
    '''
    def predict(self, query, k):

        #query = query.to(utils.device)
        if isinstance(query, np.ndarray):
            query = torch.from_numpy(query).to(utils.device)
        
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

def eval_kmeans(queries, centers, codes):
    centers_norms = torch.sum(centers ** 2, dim=0).view(1, -1)
    
    queries_norms = torch.sum(queries ** 2, dim=1).view(-1, 1)
    distances = torch.mm(queries, centers)
    distances *= -2.0
    distances += queries_norms
    distances += centers_norms
    codes = codes.to(device_cpu)
    #counts of points per center. To compute # of candidates.
    cnt = torch.zeros(num_centers, dtype=torch.long)
    bins = [[]] * num_centers
    
    for i in range(num_points):
        cnt[codes[i]] += 1  #don't recompute!!
        bins[codes[i]].append(i)
        
    num_queries = answers.size()[0]
    for num_probes in range(1, num_centers + 1):
        #ranking of indices to nearest centers
        _, probes = torch.topk(distances, num_probes, dim=1, largest=False)
        probes = probes.to(device_cpu)
        total_score = 0
        total_candidates = 0
        for i in range(num_queries):
            candidates = []
            #set of predicted bins
            tmp = set()
            for j in range(num_probes):
                candidates.append(cnt[probes[i, j]])
                tmp.add(int(probes[i, j]))
            overall_candidates = sum(candidates)
            score = 0
            for j in range(k):
                if int(codes[answers[i, j]]) in tmp:
                    score += 1
            total_score += score
            total_candidates += overall_candidates 
        print(num_probes, float(total_score) / float(k * num_queries), float(total_candidates) / float(num_queries))

'''
Input:
-dataset
Returns:
-centers. MUST ensure num_centers < len(dataset)
-codes.
'''
def build_kmeans(dataset, num_centers):
    num_points = dataset.size()[0]
    if num_centers > num_points:
        print('WARNING: num_centers > num_points! Setting num_centers = num_points')
        num_centers = num_points
    dimension = dataset.size()[1]
    centers = torch.zeros(num_centers, dimension, dtype=torch.float).to(device)
    used = torch.zeros(num_points, dtype=torch.long)
    for i in range(num_centers):
        while True:
            cur_id = random.randint(0, num_points - 1)
            if used[cur_id] > 0:
                continue
            used[cur_id] = 1
            centers[i] = dataset[cur_id]
            break
    centers = torch.transpose(centers, 0, 1)
    new_centers = torch.zeros(num_centers, dimension, dtype=torch.float).to(device)
    cnt = torch.zeros(num_centers, dtype=torch.float).to(device)
    all_ones = torch.ones(chunk_size, dtype=torch.float).to(device)
    if num_points % chunk_size != 0:
        all_ones_last = torch.ones(num_points % chunk_size, dtype=torch.float).to(device)
    all_ones_cnt = torch.ones(num_centers, dtype=torch.float).to(device)
    codes = torch.zeros(num_points, dtype=torch.long).to(device)
    for it in range(num_iterations):
        centers_norms = torch.sum(centers ** 2, dim=0).view(1, -1)
        new_centers.fill_(0.0)
        cnt.fill_(0.0)
        for i in range(0, num_points, chunk_size):
            begin = i
            end = min(i + chunk_size, num_points)
            dataset_piece = dataset[begin:end, :]
            dataset_norms = torch.sum(dataset_piece ** 2, dim=1).view(-1, 1)
            distances = torch.mm(dataset_piece, centers)
            distances *= -2.0
            distances += dataset_norms
            distances += centers_norms
            _, min_ind = torch.min(distances, dim=1)
            codes[begin:end] = min_ind
            new_centers.scatter_add_(0, min_ind.view(-1, 1).expand(-1, dimension), dataset_piece)
            if end - begin == chunk_size:
                cnt.scatter_add_(0, min_ind, all_ones)
            else:
                cnt.scatter_add_(0, min_ind, all_ones_last)
        
        if it + 1 == num_iterations:
            
            break
        cnt = torch.where(cnt > 1e-3, cnt, all_ones_cnt)
        new_centers /= cnt.view(-1, 1)
        centers = torch.transpose(new_centers, 0, 1).clone()
    #eval_kmeans(queries, centers, codes)
    return centers, codes

if __name__ == '__main__':
    
    dataset_numpy = np.load('dataset.npy')
    queries_numpy = np.load('queries.npy')
    answers_numpy = np.load('answers.npy')
    
    dataset = torch.from_numpy(dataset_numpy).to(device)
    queries = torch.from_numpy(queries_numpy).to(device)
    answers = torch.from_numpy(answers_numpy)
    
