
'''
Test driving various functions in this repo. Such as plotting,
computing alpha/beta for creating separating hyperplanes.
'''
import utils

import pdb

def plot_glove():
    #glove_data = utils.load_sift_data('query')
    #pdb.set_trace()
    
    for i in [1,100]:
        glove_data = utils.load_glove_data('train')
        glove_q = utils.load_glove_data('query')[:600]
        _, plt = utils.plot_dist_hist(glove_data, glove_q, i, 'glove')
    plt.clf()    
    
    for i in [1,100]:
        glove_c_data = utils.load_glove_c_data('train')
        glove_c_q = utils.load_glove_c_data('query')[:600]
        utils.plot_dist_hist(glove_c_data, glove_c_q, i, 'glove_c2')

def plot_sift_upto():
    for i in [4000]:
        glove_data = utils.load_sift_data('train')
        #means = glove_data.mean(0)
        #glove_data -= means

        #glove_data = glove_data / (glove_data**2).sum(-1, keepdim=True).sqrt()
        glove_q = utils.load_sift_data('query')[:300]
        #glove_q -= means
        utils.plot_dist_hist_upto(glove_data, glove_q, i, 'sift')
    #plt.clf()    
    
    for i in [4000]:
        glove_c_data = utils.load_sift_c_data('train')

        glove_c_q = utils.load_sift_c_data('query')[:300]
        pdb.set_trace()
        utils.plot_dist_hist_upto(glove_c_data, glove_c_q, i, 'sift_c')

def plot_glove_upto():
    for i in [500]:
        glove_data = utils.load_glove_data('train')
        glove_q = utils.load_glove_data('query')[:300]
        utils.plot_dist_hist_upto(glove_data, glove_q, i, 'glove')
    
    for i in [500]:
        glove_c_data = utils.load_glove_c_data('train')
        glove_c_q = utils.load_glove_c_data('query')[:300]
        utils.plot_dist_hist_upto(glove_c_data, glove_c_q, i, 'glove_c')

def compute_alpha_beta():
    #dataset = utils.load_sift_data('train').to(utils.device)
    dataset = utils.load_glove_data('train').to(utils.device)
    alpha, beta = utils.compute_alpha_beta(dataset, 10)
    print(alpha, beta)
    pdb.set_trace()

def compute_degrees_distr():
    dataset = utils.load_sift_data('train').to(utils.device)
    #dataset = utils.load_glove_data('train').to(utils.device)
    distr = utils.compute_degree_distr(dataset, 10)
    print(distr[:30])
    pdb.set_trace()

    
if __name__ == '__main__':
    #plot_sift_upto()
    #plot_glove_upto()
    #compute_alpha_beta()
    compute_degrees_distr()
