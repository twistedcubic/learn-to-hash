
'''
Pipeline to:
-create knn graphs from dataset.
-recursively partitions dataset using KaHIP in parallel.
-learn tree of neural networks in tandem with building partitions tree.
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
from model import train
from data import DataNode
import utils
from collections import defaultdict
import multiprocessing as mp
import kmeans
import logreg

import pdb

graph_file = create_graph.graph_file

data_dir = utils.data_dir
parts_path = osp.join(data_dir, 'partition')
dsnode_path = osp.join(data_dir, 'ds_node')

'''
Rerun kahip for every node/on every subtree.
Need new graph. 
Input:
-branching_l, list of indices, branching in tree so far, used for memoizing partition results.
'''
def run_kahip(graph_path, datalen, branching_l, height, opt):

    #num_parts = int(sys.argv[2])
    n_class = opt.n_class

    if n_class < 2:
        raise Exception('wrong number of parts: {}. Should be greater than or equal to 2.'.format(n_class))

    kahip_config = 'strong'
    kahip_config = opt.kahip_config

    #if configuration != 'fast' and configuration != 'eco' and configuration != 'strong':
    if kahip_config not in ['fast', 'eco', 'strong']:
        raise Exception('configuration not supported')
    
    #output_path = ' '+str(n_class)
    
    if datalen < n_class:
        n_class = datalen
        
    branching_l_len = len(branching_l)
    #if True or branching_l_len == 1:

    #parts_path = opt.parts_path_root + str(n_class) + str(kahip_config) + '{}'.format(opt.dataset_name)+''.join(branching_l) + 'ht' + str(height) + '_{}'.format('50') #'sub10')#opt.k_graph)
    parts_path = opt.parts_path_root + str(n_class) + '{}'.format(opt.dataset_name)+''.join(branching_l) + 'ht' + str(height) + '_{}_{}'.format(opt.k_graph, opt.k)

    #else:
    #    parts_path = opt.parts_path_root + str(n_class) + str(kahip_config)
        
    if opt.glove and (branching_l_len == 1):
        #if glove top level, use precomputed partition
        parts_path = utils.glove_top_parts_path(opt.n_clusters, opt)
    elif opt.sift and (branching_l_len == 1):
        #if glove top level, use precomputed partition
        parts_path = utils.sift_top_parts_path(opt.n_clusters, opt) 
    elif opt.prefix10m and (branching_l_len == 1):
        #if glove top level, use precomputed partition
        parts_path = utils.prefix10m_top_parts_path(opt.n_clusters, opt)      
    elif (branching_l_len > 1 or not os.path.exists(parts_path)):        
        #cmd = "LD_LIBRARY_PATH=./KaHIP/extern/argtable-2.10/lib ./KaHIP/deploy/kaffpa " + graph_file + " --preconfiguration=" + configuration + " --output_filename=" + output_file + " --k=" + str(num_parts)   
        cmd = os.path.join(utils.kahip_dir, "deploy", "kaffpa") + ' ' + graph_path + " --preconfiguration=" + kahip_config + " --output_filename=" + parts_path + " --k=" + str(n_class) #+ " --imbalance=" + str(3)
        pdb.set_trace()
        if os.system(cmd) != 0:
            raise Exception('Kahip error')

        #raise exception here if just want partitioning of top level
        print('parts path', parts_path)
        raise Exception('done partitioning!', parts_path)

    return parts_path

'''
create data node by reading the graph partition file.
Input:
-path: path to kahip
-ds_idx: LongTensor of indices in (entire) dataset
-height: level of current node, root has highest height.
-classes: classes in partitioning result. Int
Returns:
-datanode created from 

Use output from neural net
'''#dataset, all_ranks, ds_idx, train_node, idx2bin, height-1, branching_l, classes, opt)
def add_datanode_children(dataset, all_ranks_data, ds_idx, parent_train_node, idx2bin, height, branching_l, classes, ht2cutsz, cur_tn_idx, opt, ds_idx_ranks, toplevel=None, root=False):
    
    all_ranks, idx2weights = all_ranks_data
    n_class = opt.n_class if opt.n_class <= len(ds_idx) else len(ds_idx)
    '''
    For 2nd level, SIFT, say 64 parts, beyond 25 epochs train does not improve much.
    '''
    if opt.glove or opt.sift:
        n_epochs = 18 if len(branching_l)==1 else 15 #44 opt.n_epochs #opt.n_epochs ################stopping mechanism 65. 18 if len(branching_l)==1 else 15 <-for MCCE loss  #glove+sift: 18 then 15
    else:
        n_epochs = 18 if len(branching_l)==1 else 10 #opt.n_epochs #opt.n_epochs ################stopping mechanism 65.
    #85 good top level epoch number for MNIST. #glove+sift: 18 then 10
    
    toplevel = toplevel if toplevel is not None else (True if height > 0 else False)
    #need to train and get children idx (classes) from net.
    train_node = train.TrainNode(n_epochs, opt, height, toplevel=toplevel)
    #append node to parent
    parent_train_node.add_child(train_node, cur_tn_idx)
    
    dataset_data = dataset[ds_idx]
    if False and opt.sift:
        #'n' stands for neural and normalized
        dataset_n = dataset / dataset.norm(dim=1, p=2, keepdim=True).clamp(1)
        dataset_data_n = dataset_n[ds_idx]
    else:
        dataset_n = dataset
        dataset_data_n = dataset_data
        
    #height is 0 for leaf level nodes
    if False and height < 1:#not opt.compute_gt_nn: #height < 1: #not opt.compute_gt_nn:     True or
                
        train_node.train(dataset, dsnode, idx2bin, height)
        model = train_node.model        
        model.eval()
        classes_l = []
        chunk_sz = 90000
        
        dataset_len = len(dataset_data)
        for i in range(0, dataset_len, chunk_sz):
            
            end = min(i+chunk_sz, dataset_len)            
            cur_data = dataset_data[i:end, :]            
            classes_l.append(torch.argmax(model(cur_data), dim=1))

        classes = torch.cat(classes_l)

    action = opt.level2action[height]
    if action == 'km':
        #bottom level, use kmeans
        train_node.model = None
        train_node.trained = True
        train_node.idx2bin = idx2bin         
        
        solver = kmeans.FastKMeans(dataset_data, n_class, opt)
        d_cls_idx = solver.predict(dataset_data, k=1)

        d_cls_idx = d_cls_idx.reshape(-1)
        
        classes = torch.LongTensor(d_cls_idx)
        
        train_node.kmsolver = solver
        
        d_idx_l = [np.where(d_cls_idx==i)[0] for i in range(n_class)]
        
        train_node.probe_count_l = [len(l) for l in d_idx_l] #[(classes == i).sum().item() for i in range(n_class) ]        
    else:
        classes = torch.LongTensor(classes)
        
        if action == 'train':
            device = dataset.device
            '''
            #compute the ranks of top  classes. Using centers of all points in a class
            
            sums = torch.zeros(n_class, dataset_data.size(-1), device=device)
            classes_exp = classes.unsqueeze(1).expand_as(dataset_data).to(device)
            sums.scatter_add_(0, classes_exp, dataset_data)
            
            lens = torch.zeros(n_class)#, dtype=torch.int64)
            lens_ones = torch.ones(dataset_data.size(0))# , dtype=torch.int64)
            lens.scatter_add_(0, classes, lens_ones)
            lens = lens.to(device)
            centers = sums / lens.unsqueeze(-1)
            
            ranks = utils.dist_rank(dataset_data, k=n_class, data_y=centers, include_self=True)
            '''
            
            dsnode = DataNode(ds_idx, classes, n_class, ranks=ds_idx_ranks)
            
            #if opt.sift:          
                #center as well?
            train_node.train(dataset_n, dsnode, idx2bin, height)
            #else:
            #    train_node.train(dataset, dsnode, idx2bin, height)
            model = train_node.model        
            model.eval()
            classes_l = []
            chunk_sz = 80000
            dataset_len = len(dataset_data_n)
            for i in range(0, dataset_len, chunk_sz):
                end = min(i+chunk_sz, dataset_len)            
                cur_data = dataset_data_n[i:end, :]            
                classes_l.append(torch.argmax(model(cur_data), dim=1))

            classes = torch.cat(classes_l)
        elif action == 'logreg':
            
            train_node.model = None
            train_node.trained = True
            train_node.idx2bin = idx2bin

            cur_path = None
            if opt.glove:
                cur_path = osp.join(utils.data_dir, 'lg_glove')
            elif opt.sift:
                cur_path = osp.join(utils.data_dir, 'lg_sift')
            
            if root and cur_path is not None:                
                if osp.exists(cur_path):
                    #deserialize
                    with open(cur_path, 'rb') as file:
                        solver = pickle.load(file)
                else:
                    #serialize
                    solver = logreg.LogReg(dataset_data, classes, opt)
                    with open(cur_path, 'wb') as file:
                        pickle.dump(solver, file)
                    
            else:
                solver = logreg.LogReg(dataset_data, classes, opt)
            
            d_cls_idx = solver.predict(dataset_data, k=1)
            d_cls_idx = d_cls_idx.reshape(-1)

            classes = torch.LongTensor(d_cls_idx)
            train_node.kmsolver = solver
            
            
            d_idx_l = [np.where(d_cls_idx==i)[0] for i in range(n_class)]

            train_node.probe_count_l = [len(l) for l in d_idx_l] 
        elif action == 'kahip':
            #kahip only
            train_node.model = None
            train_node.trained = True
            train_node.idx2bin = idx2bin         
            train_node.idx2kahip = {}

            for i, cur_idx in enumerate(ds_idx):
                train_node.idx2kahip[cur_idx.item()] = classes[i]
            
            train_node.probe_count_l = [(classes == i).sum().item() for i in range(n_class) ]
        else:
            raise Exception('Action must be either kahip km or train')
    dsnode = DataNode(ds_idx, classes, n_class)    
    #ds_idx needs to be indices wrt entire dataset.    
    #y are labels of clusters, indices 0 to num_cluster. 
    
    if height > 0: 
        #recurse based on children
        procs = []
        next_act = opt.level2action[height-1]
        parallelize = next_act in ['train', 'kahip', 'logreg'] 
        if parallelize:
            p_man = mp.Manager()
            idx2classes = p_man.dict()
        
        branching_l_l = []
        child_ds_idx_l = []
        #index of child TrainNode
        tnode_idx_l = []
        ranks_l = []
        
        for cur_class in range(n_class):
            #pick the samples having this class        
            child_ds_idx = ds_idx[classes==cur_class]
            child_branching_l = list(branching_l)
            child_branching_l.append(str(cur_class))

            if len(child_ds_idx) < opt.k:
                #create train_node without model, but with base_idx, leaf_idx etc. Need to have placeholder for correct indexing.
                child_tn = train.TrainNode(opt.n_epochs, opt, height-1)
                child_tn.base_idx = len(set(idx2bin.values()))
                child_tn.leaf_idx = [child_tn.base_idx]
                for j in child_ds_idx:
                    idx2bin[j.item()] = child_tn.base_idx
                
                child_tn.probe_count_l = [len(child_ds_idx)]
                child_tn.idx2bin = idx2bin
                train_node.add_child(child_tn, cur_class)
            else:                
                ranks, all_ranks_data, graph_path = create_data_tree(dataset, all_ranks_data, child_ds_idx, train_node, idx2bin, height, child_branching_l, ht2cutsz, opt)
                branching_l_l.append(child_branching_l)
                
                #those knn graphs for kahip are one-based, and are lists and not tensors due to weights.
                if next_act == 'train':
                    k1 = max(1, int(opt.nn_mult*opt.k))
                    ranks_l.append(utils.dist_rank(dataset[child_ds_idx], k=k1))
                else:
                    ranks_l.append([])
                if parallelize:
                    datalen = len(child_ds_idx)                
                    p = mp.Process(target=process_child, args=(ranks, graph_path, datalen, child_branching_l, height, idx2classes, len(procs), ht2cutsz, opt))
                    
                    #print('processed child process!! len {}'.format(len(cur_classes)))                
                    procs.append(p)
                    p.start()
                tnode_idx_l.append(cur_class)
                child_ds_idx_l.append(child_ds_idx)
                
        for p in procs:
            p.join()
        print('~~~~~~~~~~finished p.join. check classes_l')
        
        for i in range(len(branching_l_l )):
            if parallelize:
                classes = idx2classes[i]
            else:
                classes = None
            child_branching_l = branching_l_l[i]
            child_ds_idx = child_ds_idx_l[i]
            child_ranks = ranks_l[i]
            #create root DataNode dataset, ds_idx, parent_train_node, idx2bin, height, opt
            child_dsnode = add_datanode_children(dataset, all_ranks_data, child_ds_idx, train_node, idx2bin, height-1, child_branching_l, classes, ht2cutsz, tnode_idx_l[i], opt, child_ranks)
            dsnode.add_child(child_dsnode)
    else:
        
        train_node.base_idx = len(set(idx2bin.values()))
        train_node.leaf_idx = range(train_node.base_idx, train_node.base_idx+n_class)

        if train_node.kmsolver is not None:
            predicted = train_node.kmsolver.predict(dataset_data, k=1)
            for i, pred in enumerate(predicted):
                idx2bin[ds_idx[i].item()] = train_node.base_idx + int(pred)
        else:
            #predict entire dataset at once!
            if opt.compute_gt_nn or action == 'kahip':
                for i, data in enumerate(dataset_data):                
                    predicted = train_node.idx2kahip[ds_idx[i].item()].item()
                    idx2bin[ds_idx[i].item()] = train_node.base_idx + predicted
            elif train_node.model is not None:
                dataset_data_len = len(dataset_data_n)

                chunk_sz = 80000
                if dataset_data_len > chunk_sz:
                    pred_l = []
                    
                    for p in range(0, dataset_data_len, chunk_sz):
                        cur_data = dataset_data_n[p : min(p+chunk_sz, dataset_data_len)]    
                        pred_l.append( torch.argmax(model(cur_data), dim=1) )

                    predicted = torch.cat(pred_l)
                else:
                    predicted = torch.argmax(model(dataset_data_n), dim=1)
                for i, pred in enumerate(predicted):                    
                    #idx2bin[ds_idx[i].item()] = train_node.base_idx + train_node.leaf_idx[predicted]
                    idx2bin[ds_idx[i].item()] = train_node.base_idx + int(pred)
            else:
                raise Exception('Training inconsistency')
    return dsnode

'''
TO be run in parallel.
Input:
-classes_l: NestedList object
'''
def process_child(ranks, graph_path, datalen, branching_l, height, idx2classes, proc_i, ht2cutsz, opt):
    
    n_edges = create_graph.write_knn_graph(ranks, graph_path)
    
    parts_path = run_kahip(graph_path, datalen, branching_l, height, opt)

    lines = utils.load_lines(parts_path)
    idx2classes[proc_i] = [int(line) for line in lines]

    '''
    compute_cut_size_b = False
    if compute_cut_size_b:
        cut_sz = compute_cut_size(classes, ranks)
        ht2cutsz[height].append((cut_sz, n_edges))                
    '''

'''
create data node tree by reading the graph partition file.
To be serialized and used by TrainNode to train.
Note: ds_idx is 0-based, but ranks is 1-based
ds_idx are indices for current train node.
Returns:
-root node
'''
def create_data_tree(dataset, all_ranks_data, ds_idx, train_node, idx2bin, height, branching_l, ht2cutsz, opt):

    (all_ranks, idx2weights) = all_ranks_data

    datalen = len(data_idx)
    if datalen <= opt.k:
        return None

    #create graph from data.
    data = dataset[ds_idx]
    graph_path = os.path.join(opt.data_dir, 'graph', opt.graph_file + str(opt.n_clusters) + '_'+''.join(branching_l) + 'ht' + str(height)) 
    
    #ranks are 1-based
 
    if len(branching_l) == 1:
        #only use distance at top level of tree
        ranks = create_graph.create_knn_graph(data, k=opt.k, opt=opt) #should supply opt
        all_ranks = ranks
    else:
        assert all_ranks is not None        
        #else compute part of previous graph
        ranks = create_graph.create_knn_sub_graph(all_ranks, idx2weights, ds_idx, data, opt)
    
    return ranks, all_ranks_data, graph_path

'''
To be called for creating from root. Entry point to creating the tree.
'''
def create_data_tree_root(dataset, all_ranks, ds_idx, train_node, idx2bin, height, branching_l, ht2cutsz, opt):

    datalen = len(ds_idx)
    if datalen <= opt.k:
        return None
    graph_path = os.path.join(opt.data_dir, opt.graph_file) #'../data/knn.graph'
    
    #ranks are 1-based
    if opt.glove or opt.sift or opt.prefix10m: #and len(branching_l) == 1:

        if opt.glove:
            #custom paths
        #if opt.glove and opt.k_graph==50: #april, 50NN graph file
            #graph_path = os.path.join(opt.data_dir, 'glove50_'+opt.graph_file) #'../data/knn.graph'
            graph_path = os.path.join(opt.data_dir, opt.graph_file) #'../data/knn.graph'
            #graph_path = os.path.join(opt.data_dir, 'glove10_sub10knn.graph')
            print('graph file {}'.format(graph_path))
        parts_path = run_kahip(graph_path, datalen, branching_l, height, opt)
        print('Done partitioning top level!')
        lines = utils.load_lines(parts_path)
        classes = [int(line) for line in lines]
        
        #read in all_ranks, for partitioning on further levels.
        all_ranks, idx2weights = read_all_ranks(opt)
        if opt.dataset_name != 'prefix10m':
            k1 = max(1, int(opt.nn_mult*opt.k))
            ranks = utils.dist_rank(dataset, k=k1)
        else:
            #subtract 1 as graph was created with 1-indexing for kahip.
            ranks = torch.load('/large/prefix10m10knn.graph.pt') - 1
        #create root DataNode dataset, ds_idx, parent_train_node, idx2bin, height, opt
        dsnode = add_datanode_children(dataset, (all_ranks, idx2weights), ds_idx, train_node, idx2bin, height-1, branching_l, classes, ht2cutsz, 0, opt, ranks, toplevel=True, root=True)    
        return dsnode

    #create graph from data.
    data = dataset[ds_idx]
    if len(branching_l) == 1: #this is always the case now
        #use tree created at top level throughout the hierarchy
        ranks = create_graph.create_knn_graph(data, k=opt.k, opt=opt) #should supply opt
        all_ranks = ranks
    else:
        assert all_ranks is not None        
        #else compute part of previous graph
        ranks = create_graph.create_knn_sub_graph(all_ranks, ds_idx, data, opt)
    
    n_edges = create_graph.write_knn_graph(ranks, graph_path)
    _, idx2weights = read_all_ranks(opt, path=graph_path)
        
    #create partition from graph
    #this overrides file each iteration        
    parts_path = run_kahip(graph_path, datalen, branching_l, height, opt)

    lines = utils.load_lines(parts_path)
    classes = [int(line) for line in lines]
    
    compute_cut_size_b = False and not opt.glove
    if compute_cut_size_b:
        cut_sz = compute_cut_size(classes, ranks)
        ht2cutsz[height].append((cut_sz, n_edges))                
    
    #create root DataNode dataset, ds_idx, parent_train_node, idx2bin, height, opt
    dsnode = add_datanode_children(dataset, (all_ranks, idx2weights), ds_idx, train_node, idx2bin, height-1, branching_l, classes, ht2cutsz, 0, opt, all_ranks-1, toplevel=True, root=True)
    #Note the above all_ranks is not 5*opt.k number of nearest neighbors.
    
    return dsnode

'''
Read all ranks in from precomputed glove data.
Note these neighbors are not ranked to distance, they are 
sorted according to index.
'''
def read_all_ranks(opt, path=None):
    if opt.glove:
        graph_path = osp.join(utils.glove_dir, 'graph.txt')
    elif opt.sift:
        graph_path = osp.join(utils.data_dir, 'sift_graph_10', 'graph.txt')
    elif opt.prefix10m:
        graph_path = osp.join(utils.data_dir, 'prefix10m_graph_10.txt')       
    else:
        if path is not None:
            graph_path = path
        else:
            raise Exception('Cannot read precomputed knn graph for unknown type data')
    
    ranks = []    
    lines = utils.load_lines(graph_path)[1:]
    #tuples of 2 indices, and their weights
    idx2weights = {}
    
    for i, line in enumerate(lines, 1):
                
        cur_list = line.strip().split(' ')
        cur_ranks = []
        
        for j in range(0, len(cur_list), 2):
            neigh = int(cur_list[j])
            cur_ranks.append(neigh)
            
            neigh_weight = int(cur_list[j+1])
            tup = (i, neigh) if i < neigh else (neigh, i)
            idx2weights[tup] = neigh_weight
            
        #ensure proper k! for resulting graph
        ranks.append(cur_ranks)
    
    #ranks = torch.LongTensor(ranks)
    return ranks, idx2weights

'''
Read all ranks in from precomputed SIFT data.
Note these neighbors are not ranked to distance, they are 
sorted according to index.
'''
def read_all_ranks_siftDep(opt):
    
    graph_path = osp.join(utils.data_dir, 'sift_graph_10', 'graph.txt')
    ranks = []
    
    lines = utils.load_lines(graph_path)[1:]
    #tuples of 2 indices, and their weights
    idx2weights = {}
    
    for i, line in enumerate(lines, 1):
                
        cur_list = line.strip().split(' ')
        cur_ranks = []
        
        for j in range(0, len(cur_list), 2):
            neigh = int(cur_list[j])
            cur_ranks.append(neigh)
            neigh_weight = int(cur_list[j+1])
            tup = (i, neigh) if i < neigh else (neigh, i)
            idx2weights[tup] = neigh_weight
            
        #ensure proper k! for resulting graph
        ranks.append(cur_ranks)
    
    return ranks, idx2weights

'''
Input:
-classes: list of kahip output classes
-ranks are 1-based
Should pass in total number of edges to compute ratio!
'''
def compute_cut_size(classes_l, ranks):
    idx2class = {}
    for i, iclass in enumerate(classes_l, 1):
        idx2class[i] = iclass
    #n_class = max(classes_l) #len(set(classes_l)) <--some classes are empty for high imbalance
    #cut_mx = torch.zeros(n_class, n_class)
    #should compute matrix of cuts
    cut = 0
    #ranks tensor
    ranks_is_tensor = isinstance(ranks, torch.Tensor)
    for i, row in enumerate(ranks, 1):
        
        for j in row:
            if ranks_is_tensor:
                j = j.item()
            iclass = idx2class[i]
            jclass = idx2class[j]
            if iclass != jclass:
                cut += 1
                #cut_mx[iclass-1, jclass-1] += 1
                #cut_mx[jclass-1, iclass-1] += 1
                
    print('total cut size {}'.format(cut))
    #print('cut matrix {}'.format(cut_mx))
    return cut
    
def run_kmkahip(height_preset, opt, dataset, queryset, neighbors):
    
    k = opt.k
    print('Configs: {} \n Starting data processing and training ...'.format(opt))
    #this root node is a dummy node, since it doesn't have a trained model or idx2bin
    train_node = train.TrainNode(-1, opt, -1)
    
    swap_query_to_data = False
    if swap_query_to_data:
        print('** NOTE: Modifying queryset as part of dataset **')        
        queryset = dataset[:11000]
        #queryset = dataset
        neighbors = utils.dist_rank(queryset, k=opt.k, data_y=dataset, largest=False)        
        #dist += 2*torch.max(dist).item()*torch.eye(len(dist)) #torch.diag(torch.max(dist))
        #val, neighbors = torch.topk(dist, k=opt.k, dim=1, largest=False)        
    
    #dsnode_path = opt.dsnode_path + str(opt.n_clusters)
    #dsnode = utils.pickle_load(dsnode_path)
    
    #check if need to normalize data. Remove second conditions eventually.
    if opt.normalize_data and dataset[0].norm(p=2).item() != 1 and not opt.glove:
        print('Normalizing data ...')
        dataset = utils.normalize(dataset)
        queryset = utils.normalize(queryset)
        
    #create data tree used for training
    n_clusters = opt.n_clusters
    
    height = height_preset
    n_bins = 1
    
    ds_idx = torch.LongTensor(list(range(len(dataset))))
    print('{} height: {} level2action {}'.format(ds_idx.size(), height, opt.level2action))
    
    idx2bin = {}
    ht2cutsz = defaultdict(list) 
    #used for memoizing partition results
    branching_l = ['0']
    all_ranks = None
    
    root_dsnode = create_data_tree_root(dataset, all_ranks, ds_idx, train_node, idx2bin, height, branching_l,ht2cutsz, opt)
    print('Done creating training tree. Starting evaluation ...')

    #top node only first child node is train node.
    eval_root = train.EvalNode(train_node.children[0])

    ''' Evaluate '''
    
    with torch.no_grad():
        print('About to evaluate model! {} height: {} level2action {}'.format(ds_idx.size(), height, opt.level2action))                    
        acc, probe_count, probe_count95 = train.eval_model(eval_root, queryset, neighbors, n_bins, opt)
    
    print('cut_sizes {}'.format(ht2cutsz))
    print('Configs: {}'.format(opt))
    print('acc {} probe count {} 95th {}'.format(acc, probe_count, probe_count95))

    ''' Serialize '''
    serialize_bool = False if 'kahip' in set(opt.level2action.values()) else True
    serialize_bool = True
    if serialize_bool:
        print('Serializing eval root...')
        if opt.sift:
            data_name = 'sift'
        elif opt.glove:
            data_name = 'glove'
        elif opt.prefix10m:
            data_name = 'prefix10m'
        else:
            data_name = 'mnist'
        idx2bin = eval_root.idx2bin
        if 'logreg' in opt.level2action.values():
            serial_path = 'evalroot_{}_ht{}_{}_{}{}nn{}logreg'
        else:
            serial_path = 'evalroot_{}_ht{}_{}_{}{}nn{}'
        eval_root_path = osp.join(opt.data_dir, serial_path.format(data_name, height, n_clusters, opt.k_graph, opt.k, opt.nn_mult)) 
        eval_root_dict = {'eval_root':eval_root, 'opt':opt}
        utils.pickle_dump(eval_root_dict, eval_root_path)
        print('Done serializing {}'.format(eval_root_path))
        #dsnode_path = opt.dsnode_path + str(opt.n_clusters)
        #utils.pickle_dump(root_dsnode, dsnode_path)
    
    with open(osp.join(opt.data_dir, 'cutsz_k{}_ht{}_{}'.format(k, height, n_clusters)), 'w') as file:
        file.write(str(ht2cutsz))
        file.write('\n\n')
        file.write(str(opt))
    
if __name__ == '__main__':
    opt = utils.parse_args()
    n_cluster_l = [2, 4, 16, 32, 64, 128, 256]
    n_cluster_l = [256]
    n_cluster_l = [8] #[64] #[2] #[16]
    
    # This is now set upstream, keep here for demo purposes.
    # actions can be km, kahip, train, logreg #
    opt.level2action = {0:'km', 1:'train'} 
    opt.level2action = {0:'train', 1:'train'}         
    
    opt.level2action = {0:'logreg', 2:'logreg', 3:'logreg', 4:'logreg', 5:'logreg', 6:'logreg', 7:'logreg', 8:'logreg', 9:'logreg', 10:'logreg', 11:'logreg'}
    opt.level2action = {0:'train', 1:'train'}
    
    height_l = range(1, 9)
    height_l = [1]

    #if opt.glove:
    if opt.subsample > 1:
        dataset = utils.load_glove_sub_data('train').to(utils.device)
        queryset = utils.load_glove_data('query').to(utils.device)    
        neighbors = utils.load_glove_sub_data('answers').to(utils.device)
        opt.dataset_name = 'glove'
    elif opt.glove:
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
    elif opt.prefix10m:
        dataset = utils.load_prefix10m_data('train').to(utils.device)
        queryset = utils.load_prefix10m_data('query').to(utils.device)    
        neighbors = utils.load_prefix10m_data('answers').to(utils.device)
        opt.dataset_name = 'prefix10m'        
    else:
        dataset = utils.load_data('train').to(utils.device)
        queryset = utils.load_data('query').to(utils.device)    
        neighbors = utils.load_data('answers').to(utils.device)
        opt.dataset_name = 'mnist'
    for n_cluster in n_cluster_l:
        print('n_cluster {}'.format(n_cluster))
        opt.n_clusters = n_cluster
        opt.n_class = n_cluster
        for height in height_l:
            run_kmkahip(height, opt, dataset, queryset, neighbors)
