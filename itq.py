import numpy as np

import pdb

'''
ITQ solver class.
'''

ITQ_ITER = 50

### Aux code for generating multiple candidate bins
def gen_bitlists(b):
    if b<1:
        return []
    if b==1:
        return [np.zeros(1), np.ones(1)]
    output = []
    for x in gen_bitlists(b-1):
        l0 = np.zeros(b)
        l0[:b-1] = x
        output.append(l0)
        l1 = np.ones(b)
        l1[:b-1] = x
        output.append(l1)
    return output

def gen_sorted_bitlists(b):
    lists = gen_bitlists(b)
    lists.sort(key=lambda x:np.sum(x))
    return lists

def hamming_ints(b):
    return [int(sum([(2**i)*ll[i] for i in range(len(ll))])) for ll in gen_sorted_bitlists(b)]
###

class ITQSolver:
    '''
    Input:
    -n_cluster should be power of 2 for most efficient encoding.
    '''
    def __init__(self, dataset, n_cluster):
        n_bits = 0
        while n_cluster > 1:
            n_cluster >>= 1
            n_bits += 1
        self.n_bits = n_bits
        self.model = itq_learn(dataset, n_bits)
        
    def predict(self, query, k):
        #print(query.shape)
        predicted = itq_encode(self.model, query)
        #reshape!

        hamming_ball = hamming_ints(self.n_bits)
        output = np.zeros((query.shape[0], k))
        for q in range(query.shape[0]):
            for i in range(k):
                #output[q,i] = predicted[q] ^ hamming_ball[i]
                output[q,i] = predicted[q]
        return output

def itq_encode(itq_model, Y):
    """
    Encodes given pointset by given ITQ model.

    itq_model is a model returned by an invocation
    of itq_learn (below).

    Y is a matrix in which each row is a data points.
    It can have any row-dim, and its col-dim must be
    equal to that of the matrix X that was used to
    learn the given ITQ model.
    """
    s = itq_model[0]
    W = itq_model[1]
    R = itq_model[2]

    # Shift points
    Y = Y - s
    # Project points
    V = np.dot(Y, W)
    # Rotate points
    tildeV = np.dot(V, R)
    # Generates binary codes
    B = np.zeros(tildeV.shape, dtype=np.bool)
    B[tildeV >= 0] = True
    # Return integer codes
    return B.dot(1 << np.arange(B.shape[-1] - 1, -1, -1))


def itq_learn(X, nbits, n_iter=ITQ_ITER):
    """
    Runs ITQ... or hopes to
    X is the nxd data matrix
    """

    # Zero-center points
    s = (1./X.shape[0]) * np.sum(X,0)
    X = X - s
    
    # Preliminary dimension reduction
    A2, S, A1 = np.linalg.svd(np.dot(X.transpose(), X))
    W = A2[:,:nbits]
    V = np.dot(X, W)
    
    # Initialize random rotation
    R = np.random.randn(nbits, nbits)
    Y1, SS, Y2 = np.linalg.svd(R)
    R = Y1[:, :nbits]

    # Optimize in iterations
    for i in range(n_iter):
        tildeV = np.dot(V, R)
        B = np.ones(tildeV.shape)
        B[tildeV < 0] = -1
        Z = np.dot(B.transpose(), V)
        U2, T, U1 = np.linalg.svd(Z)
        R = np.dot(U1, U2.transpose())

    # Return model
    itq_model = [s, W, R]
    return itq_model


### MAIN ###
if __name__ == '__main__':
    """
    Sample testing code. 
    """
    N = 20
    D = 10
    Q = 5
    nbits = 3

    X = np.random.random((N,D))
    itq_model = itq_learn(X, nbits)
    print(itq_encode(itq_model, X))

    Y = np.random.random((Q,D))
    print(itq_encode(itq_model, Y))
