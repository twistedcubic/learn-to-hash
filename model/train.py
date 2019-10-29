'''
Utility methods for training and evaluating models.
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

import pdb

device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
Stores trained models' state_dict's.
And the indices of data points, and the target classes
'''
class TrainNode():
    '''
    n_epochs: number of epochs.
    toplevel: if root of the training hierarchy (or non-bottom level if more than 2 levels).
    '''
    def __init__(self, n_epochs, opt, height=1, toplevel=None):

        self.n_input = opt.n_input        
        toplevel = toplevel if toplevel is not None else (True if height > 0 else False)
                
        if height == 0 and not toplevel:
            if (opt.glove or opt.sift or opt.prefix10m):
                self.n_hidden = int(opt.n_hidden//1.3)
            else:
                self.n_hidden = opt.n_hidden//2
        else:
            self.n_hidden = opt.n_hidden
        self.n_class = opt.n_class
        self.n_epochs = n_epochs
        #self.n_epochs = 1

        self.base_idx = None
        self.leaf_idx = None
        #(self, n_input, n_hidden, num_class
                
        self.model = nn.DataParallel(main.Model(self.n_input, self.n_hidden, self.n_class, opt, toplevel=toplevel).to(device))
        self.kmsolver = None
        #some train nodes result from premature leaf branches.
        self.trained = False
        #only useful if computing ground truth accuracy for dataset points
        self.idx2kahip = None
        self.children = [None]*self.n_class

        if height > 0 or toplevel:
            lr = opt.lr        
            #milestones = [20, 30, 35, 45, 50, 55, 60, 70]
            milestones = [10, 17, 24, 31, 38, 45, 50, 55, 60, 70]
            weight_decay=10**(-4)
        else:
            lr = opt.lr
            milestones = [7, 14, 21, 30, 40, 45]
            #milestones = [2, 5, 10, 20, 30, 40, 45] 
            weight_decay=10**(-3)

        #whether to double the weight of the KaHIP bin of the current training point.
        self.double_target_bin_weight = False
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30, 35, 38, 39], gamma=0.1)
        if height > 0 or toplevel:            
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.21)
        else:
            #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=1, factor=0.4)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.31)
            
    '''
    create children nodes, create data indices for children.
    Train recursively.
    Input:
    -dataset: *entire* dataset, not current partition.
    -dsnode is node for data, has dataloader of indices of dataset, targets, and children nodes 
    '''
    def train(self, dataset, dsnode, idx2bin, height=1):
        
        #with dataset, train on indices
        trainloader = dsnode.trainloader
        valloader = dsnode.valloader
        self.model.train()
                
        dataset = dataset.to(device)
                
        y = trainloader.dataset.tensors[1]
        self.probe_count_l = [(y == i).sum() for i in range(self.n_class) ]
        
        self.idx2bin = idx2bin
                
        #print('dsnode n_class {} self.n_class {} '.format(dsnode.n_class, self.n_class))
        assert dsnode.n_class == self.n_class
        
        X = None
        
        for epoch in range(self.n_epochs):
            if X == None:
                X = []
            correct = 0
            loss_l = []
            for i, data_blob in enumerate(trainloader):
                assert len(data_blob) == 3
                ds_idx, targets, cls_distr = data_blob
                
                if i == 0:
                    X.extend(list(ds_idx))
                
                if len(ds_idx) == 1:
                    #since can't batchnorm over a batch of size 1
                    continue
                batch_sz = len(ds_idx)
                
                ds_idx = ds_idx.to(device)                
                targets = targets.to(device)
                cls_distr = cls_distr.to(device)
                
                ds = dataset[ds_idx]
                                
                predicted = self.model(ds)
                                
                #take the KL divergence between predicted and actual distributions of
                #target bins of the datapoints's neighbors (including itself). 
                #shape 1x n_class, n_class x 1                    
                pred = torch.log(predicted).unsqueeze(-1)
                loss = -torch.bmm(cls_distr.unsqueeze(1), pred).sum()
                if self.double_target_bin_weight:
                    #double the weight of the target bin of current datapoint.
                    loss += F.cross_entropy(predicted, targets)                     
                loss_l.append(loss.item())
                '''
                top_val, top_idx = torch.topk(predicted, k=self.n_class, dim=1, largest=True)
                s2 = torch.zeros_like(top_val)
                s2.scatter_add_(1, ranks, top_val)
                label = torch.ones(batch_sz, self.n_class, device=device)

                #figure out where predicted is lower-ranked than target
                range_ranks = torch.FloatTensor(range(self.n_class)).expand(batch_sz, -1).to(device)
                pred_order = torch.zeros_like(label)                    
                pred_order.scatter_add_(1, top_idx, range_ranks)
                tgt_order = torch.zeros_like(label)
                tgt_order.scatter_add_(1, ranks, range_ranks)

                label[pred_order < tgt_order] = -1
                pdb.set_trace()
                loss = self.criterion(predicted, s2, label) #+ self.criterion2(predicted, targets)
                ##loss = self.criterion2(predicted, targets)
                '''
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                correct += (predicted.argmax(dim=1)==targets).sum().item()
            cur_acc = correct/len(trainloader.dataset)
            
            val_cor = 0
            self.model.eval()
            
            with torch.no_grad():
                for i, (ds_idx, targets) in enumerate(valloader):                    
                    ds_idx = ds_idx.to(device)
                    targets = targets.to(device)
                    cur_data = dataset[ds_idx]
                    predicted = self.model(cur_data)
                    val_cor += (predicted.argmax(dim=1)==targets).sum().item()                    
                    #if predicted.item() == target:                    
                    
                val_acc = val_cor/len(valloader.dataset)
                #print('Val acc: {}'.format(val_acc))
            
            print('epoch {} loss: {} train acc: {}    val acc: {} lr: {}'.format(epoch, np.mean(loss_l), cur_acc, val_acc, self.optimizer.param_groups[0]['lr']))
            self.model.train()
            if cur_acc > 0.995:# or val_acc > 0.9:
                print('Stopping training as acc is now {}'.format(cur_acc))
                break
            if True or height > 0:
                self.scheduler.step()
            else:
                c=max(val_acc/cur_acc - 0.8, 0)
                self.scheduler.step(c)
            
            
        self.trained = True        
        print('correct {} Final acc: {}'.format(correct, cur_acc))
        
                
    def add_child(self, child_node, index):
        self.children[index] = child_node
        
    '''
    For saving, create tree of state dict with this node as root, here each eval node has state dict and children nodes
    Pickle the resulting root evalnode. 
    Returns:
    EvalNode
    '''    
    def create_eval_tree(self):
        idx2bin = {}
        return EvalNode(self)

'''
Like TrainNode, contains subset of TrainNode data fields. To be serialized.
To eval, first load_state_dict(), then call eval.
'''
class EvalNode():
    def __init__(self, train_node):#, base_idx):
        #self.state_dict = state_dict
        #self.children = children
        self.trained = train_node.trained
            
        self.child_train_nodes = train_node.children
        n_input, n_hidden, self.n_class = train_node.n_input, train_node.n_hidden, train_node.n_class
        self.idx2bin = train_node.idx2bin
        
        self.idx2kahip = train_node.idx2kahip
        
        self.probe_count_l = train_node.probe_count_l
        self.kmsolver = train_node.kmsolver
        if train_node.model is not None:
            if True:
                self.state_dict = None
                self.model = train_node.model
            else:
                self.state_dict = train_node.model.state_dict()
                #need to adjust model!! To different heights!
                self.model = nn.DataParallel(main.Model(n_input, n_hidden, self.n_class, opt).to(utils.device))
            
            self.model.eval()
        else:
            self.state_dict = None
            self.model = None
            
        self.base_idx = None
        
        self.children = []
        for train_child in self.child_train_nodes:
            if train_child is None:
                continue
            child = EvalNode(train_child)
            self.children.append(child)
            
        #if self.children == []:
        if train_node.base_idx != None:
            self.base_idx = train_node.base_idx

        if train_node.leaf_idx != None:
            self.leaf_idx = train_node.leaf_idx
            
        #base_idx meaningless if not leaf node
        #self.base_idx = base_idx
        
    def load_state_dict(self):
        if self.model is not None:            
            self.model.load_state_dict(self.state_dict)
        for child in self.children:
            child.load_state_dict()

    '''
    Evaluate query on model tree.
    '''
    def eval(self, qu, qu_idx, n_bins, pred_l, probe_count_l, opt, qu_n):
        
        if self.model is None:
            #if kmeans
            if self.kmsolver is not None:
                pred_idx = self.kmsolver.predict(qu.unsqueeze(0), k=n_bins)
                pred_idx = pred_idx.reshape(-1)
            else:
                pred_idx = [self.idx2kahip[int(qu_idx)]]
        else:
            #qu_n.unsqueeze(0)
            _, pred_idx = torch.topk(self.model(qu.unsqueeze(0)), k=n_bins, dim=1, largest=True)
            
            #flatten, since pred_idx is one nested tensor
            pred_idx = pred_idx.view(-1)
                        
        #child class
        if len(self.children) > 0:            
            for i in pred_idx:                
                child = self.children[i]
                if child.trained:
                    child.eval(qu, qu_idx, n_bins, pred_l, probe_count_l, opt, qu_n)
                else:
                    #all indices must go to one bin, since not trained                    
                    assert len(child.leaf_idx) == 1 and len(child.probe_count_l) == 1
                    pred_l.append(torch.tensor(child.base_idx))
                    probe_count_l.append(self.probe_count_l[0])
                    
        else:            
            for i in pred_idx:                
                pred_l.append(self.base_idx + i) #or
                                
                probe_count_l.append(self.probe_count_l[i])

    '''
    qu: batch, 2D tensor
    '''
    def eval_batch(self, qu, qu_idx, n_bins, pred_l, probe_count_l, opt):
        
        if self.model is None:
            #if kmeans
            if self.kmsolver is not None:
                #pred_idx = self.kmsolver.predict(qu.unsqueeze(0), k=n_bins)
                pred_idx = self.kmsolver.predict(qu, k=n_bins)
                #pred_idx = pred_idx.reshape(-1)
            else:
                pred_idx = []
                for i in qu_idx:
                    pred_idx.append(self.idx2kahip[int(i)])
        else:
            _, pred_idx = torch.topk(self.model(qu), k=n_bins, dim=1, largest=True)
            
            #pred_idx = pred_idx.view(-1)
                        
        #child class
        if len(self.children) > 0:            
            for i, cur_pred_idx in enumerate(pred_idx):
                cur_qu = qu[i].unsqueeze(0)
                cur_qu_idx = qu_idx[i]
                for j in cur_pred_idx:
                    child = self.children[j]
                    if child.trained:
                        child.eval_batch(cur_qu, [cur_qu_idx] , n_bins, pred_l, probe_count_l, opt)
                    else:                        
                        #all indices must go to one bin, since not trained                    
                        assert len(child.leaf_idx) == 1 and len(child.probe_count_l) == 1
                        pred_l.append((cur_qu_idx, torch.tensor(child.base_idx)))
                        probe_count_l.append((cur_qu_idx, self.probe_count_l[0]))
                    
        else:
            #record base and elements in pred_l
            for i, cur_pred_idx in enumerate(pred_idx):
                cur_qu_idx = qu_idx[i]
                #for i in pred_idx:
                for j in cur_pred_idx:
                    pred_l.append((cur_qu_idx, self.base_idx + j))                                
                    probe_count_l.append((cur_qu_idx, self.probe_count_l[j]))
            
            '''
            for i in pred_idx:                
                pred_l.append(self.base_idx + i) #or                                
                probe_count_l.append(self.probe_count_l[i])
            '''                     

'''
Inference function.
Input: 
-queryset.
-true nearest neighbors.
'''
def eval_model(eval_root, queryset, answers, n_bins, opt):
    acc_l = []
    all_probe_count_l = []
    k = opt.k
    idx2bin = eval_root.idx2bin
    if False and opt.sift:
        queryset_n = queryset / queryset.norm(dim=1, p=2, keepdim=True)
    else:
        queryset_n = queryset
    #n_bins = opt.n_bins
    #want bins of predicted to contain true NN, ie the neighbors
    for i, qu in enumerate(queryset):
        
        pred_l = []
        probe_count_l = []
        qu_n = queryset_n[i]
        eval_root.eval(qu, i, n_bins, pred_l, probe_count_l, opt, qu_n) #note: if supply index, use i not i+1
                
        pred_bins = set(pred_l)
        
        pred_bins = set([t.item() for t in pred_bins])
        probe_count = np.sum(probe_count_l)
        all_probe_count_l.append(probe_count)
        
        neighbors = answers[i, :k]
        
        #get bins for neigh. Must be list not set.
        ##neigh_bins = [idx2bin[n] for n in neighbors]        
        try:
            if True:
                neigh_bins = [idx2bin[n.item()] for n in neighbors]
            else:
                neigh_bins = [5]
        except Exception:
            print('Exception when trying to get idx2bin!')
            pdb.set_trace()            
        
        cur_acc = np.sum([1 if b in pred_bins else 0 for b in neigh_bins]) / len(neigh_bins)
        #if cur_acc < .5:
        #    pdb.set_trace()
        acc_l.append(cur_acc)
        
    acc = np.mean(acc_l)
    probe_count = np.mean(all_probe_count_l)
    
    #compute 95th percentile probe count
    n05 = int(len(acc_l)*.05)
    val, idx = torch.topk(torch.FloatTensor(all_probe_count_l), k=n05, dim=0, largest=True)
    probe_count95 = val[-1].item()
    
    #print('95th percentile probe count {}'.format(probe_count95))
    return acc, probe_count, probe_count95


'''
Inference function wrapper: Deserialize model tree and evaluate on input data.
Input:
-tensors for query and their answers
'''
def deserialize_eval(queryset, answers, height, n_clusters, n_bins, opt):
    
    k = opt.k
    
    #replace with function!
    if opt.glove:
        data_name = 'glove'
    elif opt.sift:
        data_name = 'sift'
    elif opt.prefix10m:
        data_name = 'prefix10m'
    else:
        data_name = 'mnist'
    #eval_root = utils.pickle_load(osp.join(opt.data_dir, 'evalroot_{}_ht{}_{}svm'.format(data_name, height, n_clusters)))
    eval_root = utils.pickle_load(osp.join(opt.data_dir, 'evalroot_{}_ht{}_{}_{}{}nn{}'.format(data_name, height, n_clusters, opt.k_graph, opt.k, opt.nn_mult)))
    eval_root = eval_root['eval_root']
    ##opt = eval_root['opt']
    
    ####eval_root.load_state_dict()
    with torch.no_grad():        
        acc, probe_count, probe_count95 = eval_model(eval_root, queryset, answers, n_bins, opt)
    print('acc {} probe count {} count95: {}'.format(acc, probe_count, probe_count95))
    return acc, probe_count, probe_count95
    
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
    n_bins_l = list(range(1, 10, 2)) #[1]
    n_clusters_l = [64]#[16] #[2]
        
    acc_mx = np.zeros((len(n_clusters_l), len(n_bins_l)))
    probe_mx = np.zeros((len(n_clusters_l), len(n_bins_l)))
    probe95_mx = np.zeros((len(n_clusters_l), len(n_bins_l)))
    col_max = 0    
    for i, n_clusters in enumerate(n_clusters_l):
        for j, n_bins in enumerate(n_bins_l):  
            acc, probe_count, probe_count95  = deserialize_eval(queryset, neighbors, height, n_clusters, n_bins, opt)
            
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
            res_path = osp.join('results', 'glove_train.md')
        elif opt.sift:
            res_path = osp.join('results', 'sift_train.md')
        else:
            res_path = osp.join('results', 'mnist_train.md')
        with open(res_path, 'a') as file:
            file.write('\n\n{} **Training. MLCE. {} neighbors, k: {}, height: {}** \n\n'.format(str(date.today()), opt.nn_mult*opt.k, opt.k, height ))
            file.write(acc_md)
                                                        
def eval_deprecated():
    '''partially deprecated'''
    k = opt.k
    train_node = TrainNode(opt)
    idx2bin = {}
    dataset = utils.load_data('query') ### use dataset eventually
    dataset = dataset.to(device)
    dsnode_path = opt.dsnode_path + str(opt.n_clusters)
    #print('dsnode path {}'.format(dsnode_path))
    dsnode = utils.pickle_load(dsnode_path)
    print('dsnode {}'.format(dsnode))
    
    train_node.train(dataset, dsnode, idx2bin)
    #idx (of query) in entire dataset, bin is idx of leaf bin.
    
    eval_root = train_node.create_eval_tree()
    idx2bin = eval_root.idx2bin
    #eval root should contain dict for answers set indices and bin #, for evaluation.
        
    #serialize
    print('train.py - serializing model evaluation tree...')
    eval_root_path = osp.join(opt.data_dir, 'model_eval_root') ###########
    utils.pickle_dump(eval_root, eval_root_path)

    ## evaluate ##    
    queryset = utils.load_data('query')
    neighbors = utils.load_data('answers')
    acc, probe_count = eval_model(eval_root, queryset, neighbors, opt)
    print('train.py - Query set prediction acc {} probe count {}'.format(acc, probe_count))

