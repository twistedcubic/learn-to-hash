
'''
Demo for running k-means.
'''
import utils
from workflow_learn_kmeans import run_main, load_data, KNode
 
if __name__ == '__main__':
    opt = utils.parse_args()

    ds, qu, neigh = load_data(utils.data_dir, opt)
        
    #height_l = range(3, 11)
    height_l = [1]
    for height in height_l:
        run_main(height, ds, qu, neigh, opt)
