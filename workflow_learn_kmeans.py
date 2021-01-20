
'''
Main pipeline for running unsupervised learning methods: k-means, PCA, ITQ, random projection, or any additional method that implements a given interface.
Precise configuration can be adjusted in utils.py.
'''

import os, sys, math
import os.path as osp
import numpy as np
import sklearn
from sklearn.cluster import MiniBatchKMeans, KMeans
import pca
import cplsh
import utils
import pickle
import pdb
import time
import json
from collections import defaultdict
import kahip_solver
import kmeans
from datetime import date
import itq

import torch

data_dir = 'data'
#mbkm: minibatch k-means, km: k-means.
km_method = 'km'
max_loyd = 50

class KNode():
        def __init__(self, d_idx, dataset, n_clusters, height, ds2bins, ht2cutsz, opt):
                #d_idx are list of indices of current data in overall dataset
                self.d_idx = d_idx
                
                self.bin_idx = None
                self.ds2bins = ds2bins
                
                self.n_clusters = n_clusters
                self.height = height
                
                if height > 0:
                        self.create_child_nodes(dataset, ds2bins, ht2cutsz, opt)
                if height == 0 or len(self.children) == 0:
                        #leaf node
                        self.bin_idx = len(set(ds2bins.values()))
                        opt.bin2len_all[self.bin_idx] = len(d_idx)
                        for idx in self.d_idx:
                                ds2bins[idx] = self.bin_idx
                
        def create_child_nodes(self, dataset, ds2bins, ht2cutsz, opt):

                self.children = []
                if len(self.d_idx) < self.n_clusters:
                        if len(self.d_idx) <= 1:                        
                                return
                        if opt.cplsh:
                                self.n_clusters = 2**int(np.log(len(self.d_idx))/np.log(2))
                        else:
                                self.n_clusters = len(self.d_idx)
                
                ds = dataset[self.d_idx]
                #qu = queries[self.q_idx]
                
                child_d_idx_l, self.solver = k_means(ds, self.d_idx, ht2cutsz, self.height, self.n_clusters, opt)
                #self.d_idx2dist = {self.d_idx[i] : self.d_dist_idx[i] for i in range(len(self.d_idx)) }
                #self.q_idx2dist = {self.q_idx[i] : self.q_dist_idx[i] for i in range(len(self.q_idx)) }
                                
                for i in range(self.n_clusters):                        
                        d_idx = self.d_idx[child_d_idx_l[i]]
                        #q_idx = self.q_idx[child_q_idx_l[i]]                        
                        node = KNode(d_idx, dataset, self.n_clusters, self.height-1, ds2bins, ht2cutsz, opt)
                        self.children.append(node)

'''
Input:
-dataset: dataset for current KNode.
-dataset_idx: indices in entire dataset for current dataset/partition/KNode.
'''
def k_means(dataset, dataset_idx, ht2cutsz, height, n_clusters, opt): #ranks
        
        num_points = dataset.shape[0]

        dimension = dataset.shape[1]
        
        use_kahip_solver = False
        if opt.kmeans_use_kahip_height == height:
                use_kahip_solver = True
        if use_kahip_solver:
                solver = kahip_solver.KahipSolver()
        elif opt.fast_kmeans:                
                solver = kmeans.FastKMeans(dataset, n_clusters, opt)
        elif opt.itq:                
                solver = itq.ITQSolver(dataset, n_clusters)
        elif opt.cplsh:   
                solver = cplsh.CPLshSolver(dataset, n_clusters, opt)
        elif opt.pca:
                assert n_clusters == 2
                solver = pca.PCASolver(dataset, opt)
        elif opt.st:
                assert n_clusters == 2                
                solver = pca.STSolver(dataset, opt.glob_st_ranks, dataset_idx, opt)                
        elif opt.rp:
                if n_clusters != 2:
                        raise Exception('n_cluster {} must be 2!'.format(n_clusters))
                solver = pca.RPSolver(dataset, opt)        
        elif km_method == 'km':
                solver = KMeans(n_clusters=n_clusters, max_iter=max_loyd)
                solver.fit(dataset)
        elif km_method == 'mbkm':
                solver = MiniBatchKMeans(n_clusters=n_clusters, max_iter=max_loyd)
                solver.fit(dataset)
        else:
                raise Exception('method {} not supported'.format(km_method))
        
        
        #print("Ranking clusters for data and query points...")
        #dataset_dist = solver.transform(dataset) #could be useful, commented out for speed
        #queries_dist = solver.transform(queries)

        #the distances to cluster centers, ranked smallest first.
        #dataset_dist_idx = np.argsort(dataset_dist, axis=1)
        #queries_dist_idx = np.argsort(queries_dist, axis=1)
        
        if use_kahip_solver:
                #output is numpy array                
                d_cls_idx = solver.predict(dataset_idx)
        elif isinstance(solver, kmeans.FastKMeans):                
                d_cls_idx = solver.predict(dataset, k=1)                
                d_cls_idx = d_cls_idx.reshape(-1)
        elif isinstance(solver, cplsh.CPLshSolver):                
                                
                d_cls_idx = solver.predict(dataset, k=1)
        elif isinstance(solver, itq.ITQSolver):                
                d_cls_idx = solver.predict(dataset, k=1)  
                d_cls_idx = d_cls_idx.reshape(-1)
        else:                
                d_cls_idx = solver.predict(dataset)
        
        
        #lists of indices (not dataset points) for each class. Note each list element is a tuple.
        #list of np arrays
        d_idx_l = [np.where(d_cls_idx==i)[0] for i in range(n_clusters)]
        
        #q_idx_l = [np.where(q_cls_idx==i) for i in range(n_clusters)] #could be useful, commented out for speed
        
        compute_cut_sz_b = False
        if compute_cut_sz_b:
                ranks = utils.dist_rank(dataset, k=opt.k, opt=opt)        
                #ranks are assumed to be 1-based
                ranks += 1        
                cut_sz = compute_cut_size(d_cls_idx.tolist(), ranks)
                ht2cutsz[height].append(cut_sz)
        
        return d_idx_l, solver

'''
Check on nearest neighbors, which bins they land in.
against true NN
Input: n_bins
-neigh are indices in dataset, not feature vecs.
-ds buckets, bucket counts of elements in dataset.
Returns 
acc and probe
'''
def check_res_single(dsroot, qu, neigh, n_bins, ds2bins, opt):
        #check the bin of the nearest neighbor.
        acc_ar = np.zeros(len(qu))
                
        probe_ar = np.zeros(len(qu))
        probe_counts = np.zeros(len(qu))
        ds2bins = dsroot.ds2bins
        
        for i, q in enumerate(qu):                
                bin2len = {}
                targets = neigh[i]
                #will contain the bins retrieved by query.
                q_bins = set() #check_res_single2(node, q, probe_set, q_bins, n_bins=2):
              
                check_res_single2(dsroot, q, i, bin2len, q_bins, n_bins, opt)
                
                #get number of points probed
                #compare with target buckets                      
                cor = 0
                for neighbor in targets:                        
                        target_bin = ds2bins[neighbor]
                        if target_bin in q_bins:
                                cor += 1
                        #target_bins.append(ds2bins[neighbor])
                #len(targets) is k
                
                acc_ar[i] = cor / len(targets)
                #probe_ar[i] = sum([opt.bin2len_all[b] for b in q_bins])
                probe_ar[i] = np.array(list(bin2len.values())).sum()
                
                #compare buckets, how many of k neighbors buckets are in buckets queried                
                
        mean_probe = np.mean(probe_ar)
        acc = np.mean(acc_ar)

        n95 = int(len(qu)*.95)
        #q_nn = solver.transform(q.reshape(1,-1)).reshape(-1).argpartition(n_bins-1)[:n_bins]
        idx95 = probe_ar.argpartition(n95)[n95-1]
        probe_count95 = probe_ar[idx95]
        
        print('acc: {} mean_probe count {}'.format(acc, mean_probe))
        
        return acc, mean_probe, probe_count95

'''
Recurse down the hierarchy. 
'''
def check_res_single2(node, q, q_idx, bin2len, q_bins, n_bins, opt):
        if node.height == 0 or len(node.children) == 0:                
                bin2len[node.bin_idx] = len(node.d_idx)
                q_bins.add(node.bin_idx)
                return
        solver = node.solver
        
        #stop as soon as none of the n_bins coincide
        
        if isinstance(solver, kahip_solver.KahipSolver):
                q_nn = solver.predict([q_idx])
        elif isinstance(solver, kmeans.FastKMeans):                
                q_nn = solver.predict(q.reshape(1,-1), k=n_bins)
                q_nn = q_nn.reshape(-1)
        elif isinstance(solver, cplsh.CPLshSolver):
                #if node.height == 1:
                        #print('heit 1')
                        #pdb.set_trace()
                q_nn = solver.predict(q.reshape(1,-1), k=n_bins)
                q_nn = q_nn.reshape(-1)
        elif isinstance(solver, itq.ITQSolver):                
                q_nn = solver.predict(q.reshape(1,-1))
                q_nn = q_nn.reshape(-1)
        elif isinstance(solver, sklearn.cluster.KMeans) or isinstance(solver, sklearn.cluster.MiniBatchKMeans):
                q_nn = solver.transform(q.reshape(1,-1)).reshape(-1).argpartition(n_bins-1)[:n_bins]
        else:
                q_nn = solver.predict(q.reshape(1,-1))
        
        for bucket in q_nn:
                #pdb.set_trace()
                if bucket >= 0:
                        check_res_single2(node.children[int(bucket)], q, q_idx, bin2len, q_bins, n_bins, opt)

def load_data(data_dir, opt):
        
        if opt.glove:
                dataset = np.load(osp.join(utils.data_dir, 'glove_dataset.npy'))
                queries = np.load(osp.join(utils.data_dir, 'glove_queries.npy'))
                neigh = np.load(osp.join(utils.data_dir, 'glove_answers.npy'))
                #if DEBUG:
                #dataset = dataset[:5000]
                #neigh = utils.dist_rank(torch.from_numpy(queries).to(utils.device), k=10, data_y=torch.from_numpy(dataset).to(utils.device), opt=opt).cpu().numpy()
                
        elif opt.glove_c:
                dataset = np.load(osp.join(utils.data_dir, 'glove_c0.08_dataset.npy'))
                queries = np.load(osp.join(utils.data_dir, 'glove_c0.08_queries.npy'))
                neigh = np.load(osp.join(utils.data_dir, 'glove_answers.npy'))
                print('data loaded from {}'.format(osp.join(utils.data_dir, 'glove_c_dataset.npy')))
        elif opt.sift:
                dataset = np.load(osp.join(utils.data_dir, "sift_dataset_unnorm.npy"))
                queries = np.load(osp.join(utils.data_dir, "sift_queries_unnorm.npy"))
                neigh = np.load(osp.join(utils.data_dir, "sift_answers_unnorm.npy"))
 
        elif opt.sift_c:
                dataset = np.load(osp.join(utils.data_dir, 'sift_c_dataset.npy'))
                queries = np.load(osp.join(utils.data_dir, 'sift_c_queries.npy'))
                neigh = np.load(osp.join(utils.data_dir, 'sift_answers_unnorm.npy'))
                print('data loaded from {}'.format(osp.join(utils.data_dir, 'sift_c_dataset_unnorm.npy')))
        elif opt.prefix10m:
                dataset = np.load(osp.join(utils.data_dir, 'prefix10m_dataset.npy'))
                queries = np.load(osp.join(utils.data_dir, 'prefix10m_queries.npy'))
                neigh = np.load(osp.join(utils.data_dir, 'prefix10m_answers.npy'))
                print('data loaded from {}'.format(osp.join(utils.data_dir, 'prefix')))
        else:
                # Load MNIST data
                npy_dataset_file_name = os.path.join(data_dir, "dataset_unnorm.npy")
                npy_queries_file_name = os.path.join(data_dir, "queries_unnorm.npy")
                npy_neigh_file_name = os.path.join(data_dir, "answers_unnorm.npy")

                dataset = np.load(npy_dataset_file_name)
                queries = np.load(npy_queries_file_name)
                neigh = np.load(npy_neigh_file_name)
        
        if opt.normalize_data:
                dataset = utils.normalize_np(dataset)
                queries = utils.normalize_np(queries)        
                        
        return dataset, queries, neigh

'''
Save the tree.
'''
def save_data(dsroot):
        print("Saving results...")

        with open(os.path.join(data_dir, "kmeans_dsroot"), "wb") as output:
                pickle.dump(dsroot, output)

        print("Done.")
        
def run_kmeans(ds, qu, neigh, n_bins, n_clusters, height, ht2cutsz, opt):
        
        #used if evaluating performance on training set
        swap_query_to_data = False
        if swap_query_to_data:                
                qu = ds                
                #nearest neighbor not itself
                dist = utils.l2_dist(ds)
                dist +=  2*torch.max(dist).item()*torch.eye(len(ds))
                val, neigh = torch.topk(dist, k=opt.k, dim=1, largest=False)
                neigh = neigh.numpy()
        
        if opt.sift:
                kmeans_path = os.path.join(data_dir, 'sift', 'sift_dsroot{}ht{}'.format(n_clusters, height))
        elif opt.glove:
                if opt.fast_kmeans:
                        kmeans_path = os.path.join(data_dir, 'kmeans', 'fastkmeans_dsroot{}{}{}_{}'.format(n_clusters, km_method, max_loyd, height))
                else:
                        kmeans_path = os.path.join(data_dir, 'kmeans', 'kmeans_dsroot{}{}{}_{}'.format(n_clusters, km_method, max_loyd, height))
        elif opt.glove_c:
                #if opt.fast_kmeans:
                kmeans_path = os.path.join(data_dir, 'kmeans_glove_c', 'fastkmeans_dsroot{}{}{}_{}'.format(n_clusters, km_method, max_loyd, height))
        elif opt.sift_c:
                #if opt.fast_kmeans:
                kmeans_path = os.path.join(data_dir, 'kmeans_sift_c', 'fastkmeans_dsroot{}{}{}_{}'.format(n_clusters, km_method, max_loyd, height))
        
        else:                
                if opt.fast_kmeans:
                        kmeans_path = os.path.join(data_dir, 'kmeans_mnist', 'fastkmeans_dsroot{}{}{}_{}'.format(n_clusters, km_method, max_loyd, height))
                else:
                        kmeans_path = os.path.join(data_dir, 'kmeans_mnist', 'kmeans_dsroot{}{}{}_{}'.format(n_clusters, km_method, max_loyd, height))
        save_data = True #True
        if os.path.exists(kmeans_path) and not (opt.pca or opt.rp or opt.st): #False and
                with open(kmeans_path, 'rb') as file:
                        root = pickle.load(file)
        elif opt.cplsh and hasattr(opt, 'cplsh_root'):
                #can't serialize cpp object
                root = opt.cplsh_root
        else:
                print("Building ...")                
                d_idx = np.array(list(range(len(ds))))
                #q_idx = np.array(list(range(len(qu))))
                
                #dataset element indices to bin indices
                ds2bins = {}
                root = KNode(d_idx, ds, n_clusters, height, ds2bins, ht2cutsz, opt)
                if save_data:
                        if opt.cplsh:
                                opt.cplsh_root = root                        
                        elif not (opt.rp or opt.pca or opt.st):   
                                with open(kmeans_path, "wb") as output:
                                        pickle.dump(root, output)
                                opt.saved_path = kmeans_path
        
        acc, probe, probe95 = check_res_single(root, qu, neigh, n_bins, root.ds2bins, opt)
        print('n_clusters: {} n_bins: {} height: {} acc: {} probe: {} probe95: {}'.format(n_clusters, n_bins, height, acc, probe, probe95))
        return acc, probe, probe95
        
def run_main(height_preset, ds, qu, neigh, opt):
                
        if height_preset == 1:
                n_clusters_l = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]#, 16384, 32768, 60000] #65536]
                n_clusters_l = [1<<16]
                n_clusters_l = [16, 256] #[16]
                n_clusters_l = [16]
                #n_clusters_l = [1<<8]
        elif height_preset == 2:
                n_clusters_l = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] #2
                n_clusters_l = [16, 256] #[16]
                n_clusters_l = [256]
        elif height_preset == 3:
                n_clusters_l = [2, 4, 8, 16, 32, 64]
                n_clusters_l = [2]
        elif height_preset in range(11):
                n_clusters_l = [2]
        else:
                raise Exception('No n_clusters for height {}'.format(height_preset))
        
        print('HEIGHT: {} n_clusters: {}'.format(height_preset, n_clusters_l))
        
        #if height_preset != 1 and opt.itq:
        #        raise Exception('Height must be 1 if using ITQ')
        
        force_height = True
        
        k = opt.k
        n_repeat = opt.n_repeat_km
        n_repeat = 1
        neigh = neigh[:, 0:k]
        ht2cutsz = defaultdict(list)
        
        #acc_mx = np.zeros((len(n_clusters_l), len(n_bins_l)))
        #probe_mx = np.zeros((len(n_clusters_l), len(n_bins_l)))
        n_clusters_l_len = len(n_clusters_l)
        acc_mx = [[] for i in range(n_clusters_l_len)]
        probe_mx = [[] for i in range(n_clusters_l_len)]
        probe95_mx = [[] for i in range(n_clusters_l_len)]
        max_bin_count = 0
        start_time = time.time()
        serial_data = {}
        serial_data['k'] = k

        if opt.pca or opt.rp or opt.itq or opt.st:
                #only 1-bin probe makes sense in these settings
                opt.max_bin_count = 1
        for i, n_clusters in enumerate(n_clusters_l):
                if force_height:
                        height = height_preset
                        serial_data['height'] = height
                else:
                        height = math.floor(math.log(len(ds), n_clusters))
                bin_count = 40 #1
                               
                acc = 0
                probe = 0
                #if opt.itq or opt.pca or opt.rp:
                #        #only 1-bin probe makes sense in these settings
                #        opt.max_bin_count = 1
                        
                #keep expanding number of bins until acc reaches e.g. 0.97                                        
                while acc < opt.acc_thresh and bin_count <= min(n_clusters, opt.max_bin_count):
                        acc = 0
                        probe = 0
                        probe95 = 0
                        for l in range(n_repeat):                                
                                cur_acc, cur_probe, cur_probe95 = run_kmeans(ds, qu, neigh, bin_count, n_clusters, height, ht2cutsz, opt)
                                acc += cur_acc
                                probe += cur_probe
                                probe95 += cur_probe95
                        acc /= n_repeat
                        probe /= n_repeat
                        probe95 /= n_repeat
                                        
                        #bin_count += 1
                        bin_count += 1
                        acc_mx[i].append(acc)
                        probe_mx[i].append(probe)
                        probe95_mx[i].append(probe95)
                	
                max_bin_count = max(max_bin_count, bin_count-1)
        end_time = time.time()
        serial_data['acc_mx'] = acc_mx
        serial_data['probe_mx'] = probe_mx
        serial_data['max_loyd'] = max_loyd
        serial_data['km_method'] = km_method
        serial_data['ht2cutsz'] = ht2cutsz

        print_output = True

        if print_output:
                print('total computation time: {} hrs'.format((end_time-start_time)/3600))
                print('acc {}'.format(acc_mx))
                print('probe count {}'.format(probe_mx))
                print('ht2cutsz {}'.format(ht2cutsz))
        
        row_label = ['{} clusters'.format(i) for i in n_clusters_l]
        
        col_label = ['{} bins'.format(i+1) for i in range(max_bin_count)]
        acc_mx0 = acc_mx
        probe_mx0 = probe_mx
        probe95_mx0 = probe95_mx
        acc_mx = np.zeros((n_clusters_l_len, max_bin_count))
        probe_mx = np.zeros((n_clusters_l_len, max_bin_count))
        probe95_mx = np.zeros((n_clusters_l_len, max_bin_count))
        
        for i in range(len(n_clusters_l)):
                for j in range(len(acc_mx0[i])):
                        acc_mx[i][j] = acc_mx0[i][j]
                        probe_mx[i][j] = probe_mx0[i][j]
                        probe95_mx[i][j] = probe95_mx0[i][j]
        #[acc_mx[i][j] = acc_mx0[i][j] for j in range(len(acc_mx0[i])) for i in range(len(n_clusters_l))]
        #[probe_mx[i][j] = probe_mx0[i][j] for j in range(len(probe_mx0[i])) for i in range(len(n_clusters_l))]
        
        acc_md = utils.mxs2md([np.around(acc_mx,3), np.rint(probe_mx), np.rint(probe95_mx)], row_label, col_label)
        
        cur_method = 'k-means'
        if opt.pca:
                cur_method = 'PCA Tree'
        elif opt.st:
                cur_method = 'ST'  
        elif opt.itq:
                cur_method = 'ITQ'
        elif opt.rp:
                cur_method = 'Random Projection'                
        elif opt.cplsh:
                cur_method = 'Cross Polytope LSH'                
                
        if opt.write_res: #False 
                if opt.glove:
                        res_path = os.path.join('results', 'linear2_glove.md')
                elif opt.glove_c:
                        res_path = os.path.join('results', 'linear2_glove_c.md')  
                elif opt.sift:
                        res_path = os.path.join('results', 'linear2_sift.md')                        
                elif opt.sift_c:
                        res_path = os.path.join('results', 'linear2_sift_c.md')                  
                else:
                        res_path = os.path.join('results', 'linear2_mnist.md')
                with open(res_path, 'a') as file:
                        msg = '\n\n{} **For k = {}, height {}, method {}, max_iter: {}**\n\n'.format(str(date.today()), k, height, cur_method, max_loyd)
                        if opt.itq:
                                msg = '\n\n*ITQ*' + msg                                
                        file.write(msg)
                        file.write(acc_md)                        
        if print_output:                
                print('acc_md\n {} \n'.format(acc_md))

        if opt.glove:
                pickle_path = os.path.join(data_dir, 'glove', 'kmeans_ht{}.pkl'.format(height))
                json_path = os.path.join(data_dir, 'glove', 'kmeans_ht{}.json'.format(height))
        elif opt.glove_c:
                pickle_path = os.path.join(data_dir, 'glove_c', 'kmeans_ht{}.pkl'.format(height))
                json_path = os.path.join(data_dir, 'glove_c', 'kmeans_ht{}.json'.format(height))
        elif opt.sift:
                pickle_path = os.path.join(data_dir, 'sift', 'kmeans_ht{}.pkl'.format(height))
                json_path = os.path.join(data_dir, 'sift', 'kmeans_ht{}.json'.format(height))       
        elif opt.sift_c:
                pickle_path = os.path.join(data_dir, 'sift_c', 'kmeans_ht{}.pkl'.format(height))
                json_path = os.path.join(data_dir, 'sift_c', 'kmeans_ht{}.json'.format(height))       
        else:
                pickle_path = os.path.join(data_dir, 'kmeans_ht{}.pkl'.format(height))
                json_path = os.path.join(data_dir, 'kmeans_ht{}.json'.format(height))

        if False: #march
                utils.pickle_dump(serial_data, pickle_path)
        with open(json_path, 'w') as file:
                json.dump(serial_data, file)
        
        return acc_mx, probe_mx, probe95_mx
        

if __name__ == '__main__':

        opt = utils.parse_args()

        if opt.kmeans_use_kahip_height > 0:
                print('NOTE: will use kahip solver for height {}'.format(opt.kmeans_use_kahip_height))
        
        height_l = range(2, 10)
        height_l = [2]
        #height_l = [2]
        #height_l = range(1, 9)
        #height_l = [9,10]
        opt.bin2len_all = {}
        res_l = []
        if opt.glove_c or opt.sift_c:
                res_l = ['Catalyzed data ']
        
        ds, qu, neigh = load_data(utils.data_dir, opt)
        if opt.cplsh and opt.sift:
                ds = ds / np.sqrt((ds**2).sum(-1, keepdims=True))
                qu = qu / np.sqrt((qu**2).sum(-1, keepdims=True))                
                qu = qu[:500]
                neigh = neigh[:500]
                
                
        n_repeat = 1
        #search tree
        #global glob_st_ranks
        #if glob_st_ranks is None:
        if opt.st:
                opt.glob_st_ranks = utils.dist_rank(ds, opt.k, include_self=True, opt=opt)
                torch.save(opt.glob_st_ranks, 'st_ranks_glove')
        
        for i in range(n_repeat):
                for height in height_l:
                        acc, probe, probe95 = run_main(height, ds, qu, neigh, opt)
                        res_l.append(str(height) + ' ' + ' '.join([str(acc[0,0]), str(probe[0,0]), str(probe95[0,0])]))
        res_str = '\n'.join(res_l)
        if opt.rp:
                with open(osp.join(utils.data_dir, 'rp_data_mnist.md'), 'a') as file:
                        file.write(res_str +'\n')
        
        print(res_str)
        if hasattr(opt, 'saved_path'):
                print('need to delete ', opt.saved_path)
