import sys
import h5py
import numpy as np
import struct

f = h5py.File(sys.argv[1], 'r')
print f.keys()
dataset = f['train'][:]
print dataset.shape, dataset.dtype
queries = f['test'][:]
print queries.shape, queries.dtype
answers = f['neighbors'][:]
print answers.shape, answers.dtype

def serialize(a, file_name):
    if len(a.shape) != 2:
        raise Exception('array must be two-dimensional')
    if a.dtype != np.float32 and a.dtype != np.int32:
        raise Exception('invalid dtype')
    if a.dtype == np.float32:
        spec = 'f'
    else:
        spec = 'i'
    print spec
    with open(file_name, 'wb') as output:
        output.write(struct.pack('Q', a.shape[0]))
        for i in range(a.shape[0]):
            output.write(struct.pack('Q', a.shape[1]))
            output.write(struct.pack('%d%s' % (a.shape[1], spec), *a[i]))

serialize(dataset, sys.argv[2])
serialize(queries, sys.argv[3])
serialize(answers, sys.argv[4])
