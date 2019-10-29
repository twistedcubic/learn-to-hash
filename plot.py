
'''
Utilities functions for plotting and visualization.
'''
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns

import json
import pandas as pd
import numpy as np
import utils
import os
import os.path as osp
import re
import torch

import pdb

'''
Plot kmeans results
Input:
acc_mx: list of lists

'''
def plot_kmeans(acc_mx, probe_mx, height, k, json_data, opt):
    
    df = create_df(acc_mx, probe_mx, height, k, opt) 
    method, max_loyd = json_data['km_method'], json_data['max_loyd']
    
    fig = sns.scatterplot(x='probe_count', y='acc', data=df)
    fig.set_title('height: {}, k: {}, km method: {}, max_lloyd: {}'.format(height, k, method, max_loyd))

    if opt.glove:
        fig_path = osp.join(opt.data_dir, 'glove', 'kmeans_ht{}_2.jpg'.format(height))
    else:
        fig_path = osp.join(opt.data_dir, 'kmeans_ht{}_2.jpg'.format(height))

    fig.figure.savefig(fig_path)

'''
Plot kmeans for multiple height levels.
'''
def plot_kmeans_multi(acc_mx_l, probe_mx_l, height_l, k, json_data, opt):

    df_l = []
    height_df_l = []
    for i, acc_mx in enumerate(acc_mx_l):
        probe_mx = probe_mx_l[i]
        height = height_l[i]
        df = create_df(acc_mx, probe_mx, height, k, opt)
        df_l.append(df)
        height_df_l.extend([height] * len(df))
    
    method, max_loyd = json_data['km_method'], json_data['max_loyd']
    df = pd.concat(df_l, axis=0, ignore_index=True)
    
    height_df = pd.DataFrame({'height': height_df_l})
    
    df = pd.concat([df, height_df], axis=1)
        
    fig = sns.scatterplot(x='probe_count', y='acc', hue='height', data=df)
    
    if opt.glove:
        fig.set_title('Number of candidates vs accuracy on GloVe. Height: {}, k: {}, km method: {}, max_lloyd: {}'.format(height_l, k, method, max_loyd))
        fig_path = osp.join(opt.data_dir, 'glove', 'glove_kmeans_ht{}.jpg'.format(''.join(map(str, height_l)) ))
    else:
        fig.set_title('height: {}, k: {}, km method: {}, max_lloyd: {}'.format(height_l, k, method, max_loyd))    
        fig_path = osp.join(opt.data_dir, 'kmeans_ht{}.jpg'.format(''.join(map(str, height_l)) ))
    
    fig.figure.savefig(fig_path)
    print('Figure saved to {}'.format(fig_path))
    
'''
Create dataframe for given input nested lists.
Input lists must have same shapes.
Returns:
-dataframe.
'''
def create_df(acc_mx, probe_mx, height, k, opt):
    #construct probe_count, acc, and dist_count

    #total number of points we compute distances to
    dist_count_l = []
    acc_l = []
    probe_l = []
    counter = 0
    n_clusters_ar = [2**(i+1) for i in range(20)]
    
    #i indicates n_clusters
    for i, acc_ar in enumerate(acc_mx):
        n_clusters = n_clusters_ar[i]
        #j is n_bins
        for j, acc in enumerate(acc_ar):
            probe_count = probe_mx[i][j]
            if not opt.glove and not opt.sift:
                if height == 1 and probe_count > 2000:            
                    continue
                elif probe_count > 3000:
                    continue
            
            # \sum_u^h n_bins^u * n_clusters * k
            exp = np.array([l for l in range(height)])
            
            dist_count = np.sum(k * n_clusters * j**exp)
            if not opt.glove and not opt.sift:
                if dist_count > 50000:
                    continue
            dist_count_l.append(dist_count)
            acc_l.append(acc)
            #probe_l.append(probe_count)
            probe_l.append(probe_count + dist_count)
            
            counter += 1
    
    df = pd.DataFrame({'probe_count':probe_l, 'acc':acc_l, 'dist_count':dist_count_l})
    return df
        

def plot_single_ht(height):
    
    opt = utils.parse_args()
    if opt.glove:
        data_path = osp.join(opt.data_dir, 'glove', 'kmeans_ht{}.json'.format(height))
    else:
        data_path = osp.join(opt.data_dir, 'kmeans_ht{}.json'.format(height))
    #use opt for name
    with open(data_path, 'r') as file:
        json_data = json.load(file)
        
    acc_mx = json_data['acc_mx'] 
    probe_mx = json_data['probe_mx']
    assert json_data['height'] == height
    k = json_data['k']
    
    plot_kmeans(acc_mx, probe_mx, height, k, json_data, opt)
    
def plot_multi_ht(height_l):
    
    opt = utils.parse_args()
    acc_mx_l = []
    probe_mx_l = []
    height_l = [1,2,3]
    #use opt for name
    for height in height_l:
        if opt.glove:            
            kmeans_path = osp.join(opt.data_dir, 'glove', 'kmeans_ht{}.json'.format(height))
        else:
            kmeans_path = osp.join(opt.data_dir, 'kmeans_ht{}.json'.format(height))
        with open(kmeans_path, 'r') as file:
            #assuming json_data have consistent hyperparameters
            json_data = json.load(file)
        acc_mx = json_data['acc_mx']
        acc_mx_l.append(acc_mx)
        probe_mx = json_data['probe_mx']
        probe_mx_l.append(probe_mx)
        assert json_data['height'] == height
    k = json_data['k']
    
    plot_kmeans_multi(acc_mx_l, probe_mx_l, height_l, k, json_data, opt)

'''
Plot, line plot.
Input:
-x, y: np arrays, duplicities should already be included.
'''
def acc_probe_lineplot(probe_ar, acc_ar, method_l, height, n_clusters, opt):
    if isinstance(probe_ar, torch.Tensor):
        probe_ar = probe_ar.numpy()
    if isinstance(acc_ar, torch.Tensor):
        acc_ar = acc_ar.numpy()

    if opt.glove:
        data_len = 1180000
    elif opt.sift:
        data_len = 1000000
    else:
        data_len = 60000
        
    #probe_ar = np.log(probe_ar)
    probe_max = max([int(count) for count in probe_ar])
    
    df = pd.DataFrame({'probe_count':probe_ar, 'acc':acc_ar, 'method':method_l})
    dashes = {'km':False, 'neural':True, 'km95':False, 'neural95':True}
    palette = sns.color_palette("mako_r", 2)
    palette = sns.color_palette("ch:2.5,-.2,dark=.3", n_colors=2)
    
    fig = sns.lineplot( x='acc', y='probe_count', hue='method', style='method', palette=palette, markers=True, data=df)

    if opt.glove:
        data_name = 'GloVe'
    elif opt.sift:
        data_name = 'SIFT'
    if opt.glove_c:
        data_name = 'GloVeCatalyzer'
    elif opt.sift_c:
        data_name = 'SIFTCatalyzer'
    else:
        data_name = 'MNIST'
        
    fig.set_title('Number of candidates vs accuracy on {}. m: {} on {} levels'.format(data_name, n_clusters, height))
    fig.set(xlabel='Accuracy', ylabel='Number of candidates')
    fig.set(xlim=(.5, .97))
    fig.set(ylim=(-200, probe_max+300))
    fig_path = osp.join(opt.data_dir, '{}_ht{}_{}.jpg'.format(data_name, height, n_clusters))
    
    fig.figure.savefig(fig_path)
    print('Figure saved to {}'.format(fig_path))

'''
Main driver for plotting acc vs probe_count plots
'''
def acc_probe_lineplot_main():

    opt = utils.parse_args()
    sep_patt = re.compile('[\s/]+')
    
    #data_str1 = '0.178 / 35.0  0.39 / 289.0 0.495 / 769.0 0.565 / 1465.0 0.618 / 2372.0'
    with_catalyzer = True
    #if with_catalyzer:
    #    km_data_path = osp.join(utils.data_dir, 'km_plot_data_c')
    #else:
    km_data_path = osp.join(utils.data_dir, 'km_plot_data')
    train_data_path = osp.join(utils.data_dir, 'train_plot_data')
    
    #data format example: data_str_km = '0.788 / 50803.0 0.791 / 53157.0 0.795 / 55564.0 0.798 / 58021.0'
    
    data_str_km = utils.load_lines(km_data_path)[0]
    data_str_train = utils.load_lines(train_data_path)[0]
    
    label_ar = ['km', 'neural']
    method_l = []
    acc_l = []
    probe_l = []
    #determine this dynamically
    probe95_plot_l = [True, True]

    for i, data_str in enumerate([data_str_km, data_str_train]):
        data_ar = sep_patt.split(data_str.strip())
        if probe95_plot_l[i]:
            
            assert (len(data_ar) % 3) == 0
            step = 3
        else:
            assert (len(data_ar) & 1) == 0
            step = 2
        data_ar = list(map(float, data_ar))

        acc_l.extend(data_ar[0::step])
        probe_l.extend(data_ar[1::step])
        method_l.extend([label_ar[i]]*(len(data_ar)//step))
        
        if False and probe95_plot:
            acc_l.extend(data_ar[0::step])
            probe_l.extend(data_ar[2::step])
            method_l.extend([label_ar[i]+'95']*(len(data_ar)//step))
        
    acc_ar = np.array(acc_l)
    probe_ar = np.array(probe_l)
    
    
    height = 1
    n_clusters = 256
    acc_probe_lineplot(probe_ar, acc_ar, method_l, height, n_clusters, opt)    
    
if __name__=='__main__':

    if True:
        acc_probe_lineplot_main()
    else:
        plot_single = False
        if plot_single:
            height = 2
            plot_single_ht(height)
        else:
            height_l = [1, 2, 3]
            plot_multi_ht(height_l)
