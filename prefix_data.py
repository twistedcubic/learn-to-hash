
'''
Processes/parses Deep 1B prefix dataset
'''
import _init_paths
import struct
import os.path as osp
import torch
import utils
import numpy as np

import pdb

def create_dataset():
    data_dir = 'data/prefix'
    data_dir = '/large/prefix'
    data_dir = '../deep1b.dat'
    data_dir = '/large/deep1b.dat'
    #with open(osp.join(data_dir, 'base_00'), 'rb') as file:
    with open(osp.join(data_dir), 'rb') as file:
        data = file.read() #(8*96+8)
    i = 0
    skip = 4*96 #4*97
    
    data_len = len(data)
    #last vector can be cut off in the middle of vec
    ##assert data_len % 4 == 0 and (data_len-4) % 96 == 0
    n_queries = 100000
    n_data = 10000000
    #n_queries = 1
    #n_data = 9
    n_total = n_data + 2*n_queries #2x to account for dups    
    
    data_l = []
    stop_len = data_len - skip
    counter = 0
    byte_set = set()
    dup_count = 0
    
    while i < stop_len:
        
        #cur_bytes = data[i+4:i+skip] <--if download directly 
        cur_bytes = data[i:i+skip] #
        if cur_bytes in byte_set:            
            dup_count += 1
            i += skip
            continue
        
        byte_set.add(cur_bytes)
        
        data_l.append(list(struct.unpack('96f', cur_bytes)))
        #except struct.error:
        #    print('struct error')
        #    pdb.set_trace()
        counter += 1
        if counter == n_total:
            #pdb.set_trace()
            break
        
        i += skip

    print('dup count {}'.format(dup_count))
    print('number of vectors {}'.format(len(data_l)))
    queries = torch.FloatTensor(data_l[:n_queries])
    dataset = torch.FloatTensor(data_l[n_queries : n_queries+n_data])
    torch.save(queries, '/large/prefix10m_queries.pt')
    torch.save(dataset, '/large/prefix10m_dataset.pt')
    #pdb.set_trace()
    #need bah size 200 for 10mil to not be out of memory
    answers = utils.dist_rank(queries, k=10, data_y=dataset, include_self=True)
    #pdb.set_trace()
    
'''
Input:
-2D tensor
'''
def remove_dup(data):
    
    selected = torch.ones(data.size(0), dtype=torch.uint8)
    for i, d in enumerate(data[1:], 1):
        prev = d.eq(data).sum(-1)
        #if i == 10:
        #    pdb.set_trace()
        if (prev == 96).sum().item() > 1: #1 for itself
            selected[i] = 0
            print('{} dup '.format(i))
        
    data = data[selected]
    print('{} duplicates'.format(len(data) - selected.sum()))
    np.save('data/prefix1m_dataset2.npy', data.cpu().numpy())
    
    return data
    
if __name__ == '__main__':
    remove_dup_bool = False
    if remove_dup_bool:
        dataset = np.load('data/prefix1m_dataset.npy')
        dataset = torch.from_numpy(dataset).cuda()
        dataset = remove_dup(dataset)

        queries = torch.from_numpy(np.load('data/prefix1m_queries.npy')).cuda()
        answers = utils.dist_rank(queries, k=10, data_y=dataset, include_self=True)

        np.save('data/prefix1m_answers2.npy', answers.cpu().numpy())
    else:
        create_dataset()
