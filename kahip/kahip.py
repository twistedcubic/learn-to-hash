
'''
Run Kahip with input knn graph to produce graph partition. Create DataNodes from Kahip results.
Many functionalities here are superceded by kmkahip.py.
'''
import _init_paths
import sys
import os
import os.path as osp
import create_graph
import torch
#torch.multiprocessing.set_start_method("spawn")
import numpy as np
import argparse
import utils
import math
from model import train
from data import DataNode
import utils
from collections import defaultdict
import multiprocessing as mp
#from multiprocessing import Process
import pdb

kahip_dir = utils.kahip_dir
graph_file = create_graph.graph_file

#def create ou
data_dir = create_graph.data_dir
#parts_dir = os.path.join(graph_dir, 'partition_%d_%s' % (k, configuration))
parts_path = osp.join(data_dir, 'partition')

#if not os.path.exists(parts_dir):
#    os.makedirs(parts_dir)
#parts_path = osp.join(parts_dir, 'partition.txt')
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
    parts_path = opt.parts_path_root + str(n_class) + str(kahip_config) + '_'+''.join(branching_l) + 'ht' + str(height)
    #else:
    #    parts_path = opt.parts_path_root + str(n_class) + str(kahip_config)
        
    if opt.glove and (branching_l_len == 1):
        #if glove top level, use precomputed partition
        parts_path = utils.glove_top_parts_path(opt.n_clusters)
    elif opt.sift and (branching_l_len == 1):
        #if glove top level, use precomputed partition
        parts_path = utils.sift_top_parts_path(opt.n_clusters) ##implement!!
        
    elif (branching_l_len > 1 or not os.path.exists(parts_path)):        
        #cmd = "LD_LIBRARY_PATH=./KaHIP/extern/argtable-2.10/lib ./KaHIP/deploy/kaffpa " + graph_file + " --preconfiguration=" + configuration + " --output_filename=" + output_file + " --k=" + str(num_parts)   
        cmd = os.path.join(kahip_dir, "deploy", "kaffpa") + ' ' + graph_path + " --preconfiguration=" + kahip_config + " --output_filename=" + parts_path + " --k=" + str(n_class) #+ " --imbalance=" + str(3)
        
        if os.system(cmd) != 0:
            raise Exception('Kahip error')    
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
def add_datanode_children(dataset, all_ranks_data, ds_idx, parent_train_node, idx2bin, height, branching_l, classes, ht2cutsz, opt):
    
    all_ranks, idx2weights = all_ranks_data
    #parts_path = opt.parts_path
    #DataNode(self, ds_idx, n_input, n_hidden , n_class):
    #kahip outputs lines of indices indicating cluster numbers
    
    #lines = utils.load_lines(parts_path)
    #classes = [int(line) for line in lines]
    
    #n_class = len(set(classes))
    n_class = opt.n_class #must be opt.n_class in case kahip skips classes
    #classes_o = classes
    classes = torch.LongTensor(classes)
    n_epochs = 1 if len(branching_l)==1 else 1 #opt.n_epochs ################stopping mechanism 65.
    #85 good top level epoch number for MNIST.
    
    #need to train and get children idx (classes) from net.
    train_node = train.TrainNode(n_epochs, opt, height)
    #append node to parent
    parent_train_node.add_child(train_node)
    
    dsnode = DataNode(ds_idx, classes, n_class)
    dataset_data = dataset[ds_idx]
        
    #height is 0 for leaf level nodes
    if True or height < 1:#not opt.compute_gt_nn: #height < 1: #not opt.compute_gt_nn:     True or
                
        train_node.train(dataset, dsnode, idx2bin, height)
        model = train_node.model        
        model.eval()
        classes_l = []
        chunk_sz = 10000 #set this dynamically!
        #total_chunks = (len(dataset_data)-1) // chunk_sz + 1
        dataset_len = len(dataset_data)
        for i in range(0, dataset_len, chunk_sz):
            #end = min((i+1)*chunk_sz, dataset_len)
            end = min(i+chunk_sz, dataset_len)            
            cur_data = dataset[i:end, :]            
            classes_l.append(torch.argmax(model(cur_data), dim=1))

        classes = torch.cat(classes_l)
        
    else:
        train_node.model = None
        train_node.trained = True
        train_node.idx2bin = idx2bin        
        
        train_node.idx2kahip = {}
        for i, cur_idx in enumerate(ds_idx):
            train_node.idx2kahip[cur_idx.item()] = classes[i]
            
        train_node.probe_count_l = [(classes == i).sum().item() for i in range(n_class) ]
        
    #ds_idx needs to be indices wrt entire dataset.    
    #y are labels of clusters, indices 0 to num_cluster. 
    
    if height > 0:            
        #recurse based on children
        procs = []
        #classes_l = utils.NestedList()
        p_man = mp.Manager()
        idx2classes = p_man.dict()
        
        branching_l_l = []
        child_ds_idx_l = []
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
                train_node.add_child(child_tn)
            else:
                ranks, all_ranks_data, graph_path = create_data_tree(dataset, all_ranks_data, child_ds_idx, train_node, idx2bin, height, child_branching_l, ht2cutsz, opt)
                
                datalen = len(child_ds_idx)                
                p = mp.Process(target=process_child, args=(ranks, graph_path, datalen, child_branching_l, height, idx2classes, len(procs), ht2cutsz, opt))        
                branching_l_l.append(child_branching_l)
                #print('processed child process!! len {}'.format(len(cur_classes)))

                procs.append(p)
                p.start()

                branching_l_l.append(child_branching_l)
                child_ds_idx_l.append(child_ds_idx)
                
            '''
            n_edges = create_graph.write_knn_graph(ranks, graph_path)    
            parts_path = run_kahip(graph_path, datalen, branching_l, height, opt)
            lines = utils.load_lines(parts_path)
            classes = [int(line) for line in lines]
            compute_cut_size_b = True and not opt.glove
            if compute_cut_size_b:
                cut_sz = compute_cut_size(classes, ranks)
                ht2cutsz[height].append((cut_sz, n_edges))                
            '''        
        
        for p in procs:
            p.join()
        print('~~~~~~~~~~finished p.join. check classes_l')
        
        for i in range(len(procs)):
            classes = idx2classes[i]
            child_branching_l = branching_l_l[i]
            child_ds_idx = child_ds_idx_l[i]
            #create root DataNode dataset, ds_idx, parent_train_node, idx2bin, height, opt
            child_dsnode = add_datanode_children(dataset, all_ranks_data, child_ds_idx, train_node, idx2bin, height-1, child_branching_l, classes, ht2cutsz, opt)
            dsnode.add_child(child_dsnode)
            '''
            if child_dsnode != None:                
                dsnode.add_child(child_dsnode)
            else:
                #create train_node without model, but with base_idx, leaf_idx etc. Need to have placeholder for correct indexing.
                child_tn = train.TrainNode(opt.n_epochs, opt, height-1)
                lock.acquire()
                child_tn.base_idx = len(set(idx2bin.values()))
                child_tn.leaf_idx = [child_tn.base_idx]
                for j in child_ds_idx:
                    idx2bin[j.item()] = child_tn.base_idx
                lock.release()
                child_tn.probe_count_l = [len(child_ds_idx)]
                child_tn.idx2bin = idx2bin
                train_node.add_child(child_tn)
            '''
            
    else:
        #either height == 0, or no TrainNode child was added        
        train_node.base_idx = len(set(idx2bin.values()))
        train_node.leaf_idx = range(train_node.base_idx, train_node.base_idx+n_class)
        for i, data in enumerate(dataset_data):
            
            if opt.compute_gt_nn:
                predicted = train_node.idx2kahip[ds_idx[i].item()].item()
            else:
                predicted = torch.argmax(model(data.unsqueeze(0)), dim=1).item()
            #idx2bin[ds_idx[i].item()] = train_node.base_idx + train_node.leaf_idx[predicted]
            idx2bin[ds_idx[i].item()] = train_node.base_idx + predicted
            
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
    
    compute_cut_size_b = True and not opt.glove
    if compute_cut_size_b:
        cut_sz = compute_cut_size(classes, ranks)
        ht2cutsz[height].append((cut_sz, n_edges))                


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
    
    #create graph from data.
    data = dataset[ds_idx]
    datalen = len(data)
    if datalen <= opt.k:
        return None
    graph_path = os.path.join(opt.data_dir, 'graph', opt.graph_file + str(opt.n_clusters) + '_'+''.join(branching_l) + 'ht' + str(height)) #'../data/knn.graph'
    #str(n_class) + str(kahip_config) + '_'+''.join(branching_l) + 'ht' + str(height)
    #ranks are 1-based
    '''
    if opt.glove and len(branching_l) == 1:
        parts_path = run_kahip(graph_path, datalen, branching_l, height, opt)

        lines = utils.load_lines(parts_path)
        classes = [int(line) for line in lines]
        #read in all_ranks
        all_ranks = read_all_ranks_glove(opt)
        
        #create root DataNode dataset, ds_idx, parent_train_node, idx2bin, height, opt
        dsnode = add_datanode_children(dataset, all_ranks, ds_idx, train_node, idx2bin, height-1, branching_l, classes, ht2cutsz, opt)    
        return dsnode
    '''
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
To be called for creating from root
'''
def create_data_tree_root(dataset, all_ranks, ds_idx, train_node, idx2bin, height, branching_l, ht2cutsz, opt):

    #create graph from data.
    data = dataset[ds_idx]
    datalen = len(data)
    if datalen <= opt.k:
        return None
    graph_path = os.path.join(opt.data_dir, opt.graph_file) #'../data/knn.graph'
    
    #ranks are 1-based
    if opt.glove or opt.sift: #and len(branching_l) == 1:
        parts_path = run_kahip(graph_path, datalen, branching_l, height, opt)

        lines = utils.load_lines(parts_path)
        classes = [int(line) for line in lines]
        #read in all_ranks
        if opt.glove:
            all_ranks, idx2weights = read_all_ranks_glove(opt)
        elif opt.sift:
            all_ranks, idx2weights = read_all_ranks_sift(opt) ###implement!!
        
        #create root DataNode dataset, ds_idx, parent_train_node, idx2bin, height, opt
        dsnode = add_datanode_children(dataset, (all_ranks, idx2weights), ds_idx, train_node, idx2bin, height-1, branching_l, classes, ht2cutsz, opt)    
        return dsnode
   
    if len(branching_l) == 1: #this is always the case now
        #only use distance at top level of tree
        ranks = create_graph.create_knn_graph(data, k=opt.k, opt=opt) #should supply opt
        all_ranks = ranks
    else:
        assert all_ranks is not None        
        #else compute part of previous graph
        ranks = create_graph.create_knn_sub_graph(all_ranks, ds_idx, data, opt)
    
    n_edges = create_graph.write_knn_graph(ranks, graph_path)
    
    #graph_dir = create_graph.data_dir
    #graph_file = os.path.join(graph_dir, graph_file)
    
    #create partition from graph
    #this overrides file each iteration
    #parts_path = opt.parts_path_root
    
    parts_path = run_kahip(graph_path, datalen, branching_l, height, opt)

    lines = utils.load_lines(parts_path)
    classes = [int(line) for line in lines]

    compute_cut_size_b = True and not opt.glove
    if compute_cut_size_b:
        cut_sz = compute_cut_size(classes, ranks)
        ht2cutsz[height].append((cut_sz, n_edges))                
    
    #create root DataNode dataset, ds_idx, parent_train_node, idx2bin, height, opt
    dsnode = add_datanode_children(dataset, (all_ranks, None), ds_idx, train_node, idx2bin, height-1, branching_l, classes, ht2cutsz, opt)
    
    return dsnode

'''
Read all ranks in from precomputed glove data.
Note these neighbors are not ranked to distance, they are 
sorted according to index.
'''
def read_all_ranks_glove(opt):
    ##need weights!!!!!!!!!!!!
    graph_path = osp.join(utils.glove_dir, 'normalized','knn_100','graph_10', 'graph.txt')
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
    
if __name__ == '__main__':

    opt = utils.parse_args()
    k = opt.k
    print('Configs: {}'.format(opt))
    #this root node is a dummy node, since it doesn't have a trained model or idx2bin
    train_node = train.TrainNode(-1, opt, -1)

    if opt.glove:
        dataset = utils.load_glove_data('train').to(utils.device)
        queryset = utils.load_glove_data('query').to(utils.device)    
        neighbors = utils.load_glove_data('answers').to(utils.device)   
    else:
        dataset = utils.load_data('train').to(utils.device)
        queryset = utils.load_data('query').to(utils.device)    
        neighbors = utils.load_data('answers').to(utils.device)
       
    ######uncomment
    if False:
        #dataset = queryset ##remove Dec 15
        queryset = dataset
        dist = utils.l2_dist(dataset)
        dist += 2*torch.max(dist).item()*torch.eye(len(dist)) #torch.diag(torch.max(dist))
        val, neighbors = torch.topk(dist, k=opt.k, dim=1, largest=False)        
    
    if False:
        #queryset = dataset
        dist = utils.l2_dist(queryset)
        dist += 2*torch.max(dist).item()*torch.eye(len(dist)) #torch.diag(torch.max(dist))
        val, neighbors = torch.topk(dist, k=opt.k, dim=1, largest=False)    
    
    #dsnode_path = opt.dsnode_path + str(opt.n_clusters)
    #print('dsnode path {}'.format(dsnode_path))
    #dsnode = utils.pickle_load(dsnode_path)
    #print('dsnode {}'.format(dsnode))

    #train_node.train(dataset, dsnode, idx2bin)
    #########
    #check if need to normalize data. Remove second conditions eventually.
    if opt.normalize_data and dataset[0].norm(p=2).item() != 1 and not opt.glove:
        dataset = utils.normalize(dataset)
        queryset = utils.normalize(queryset)
        
    #create data tree used for training
    n_clusters = opt.n_clusters
    if False:
        height = math.floor(math.log(len(dataset), n_clusters))
    height = 1
    n_bins = 2
    n_bins = 20
    
    ds_idx = torch.LongTensor(list(range(len(dataset))))
    print(ds_idx.size())
    #parts_path = s + str(n_clusters, )
    idx2bin = {}
    ht2cutsz = defaultdict(list) 
    #used for memoizing partition results
    branching_l = ['0']
    all_ranks = None
    #lock = Lock()
    root_dsnode = create_data_tree_root(dataset, all_ranks, ds_idx, train_node, idx2bin, height, branching_l,ht2cutsz, opt)
    
    assert len(train_node.children) == 1
    eval_root = train.EvalNode(train_node.children[0])

    '''serialize'''
    print('Serializing eval root...')
    idx2bin = eval_root.idx2bin
    eval_root_path = osp.join(opt.data_dir, 'evalroot_k{}_ht{}_{}'.format(k, height, n_clusters)) 
    eval_root_dict = {'eval_root':eval_root, 'opt':opt}
    utils.pickle_dump(eval_root_dict, eval_root_path)
    
    dsnode_path = opt.dsnode_path + str(opt.n_clusters)
    utils.pickle_dump(root_dsnode, dsnode_path)
    
    with open(osp.join(opt.data_dir, 'cutsz_k{}_ht{}_{}'.format(k, height, n_clusters)), 'w') as file:
        file.write(str(ht2cutsz))
        file.write('\n\n')
        file.write(str(opt))

    ''' Evaluate '''
    ###eval_root.load_state_dict()
    with torch.no_grad():
        print('About to evaluate model!')
        
        acc, probe_count = train.eval_model(eval_root, queryset, neighbors, n_bins, opt)

    print('cut_sizes {}'.format(ht2cutsz))
    print('Configs: {}'.format(opt))
    print('acc {} probe count {}'.format(acc, probe_count))
    
