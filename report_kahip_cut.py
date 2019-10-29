import os, sys, json
import numpy as np
from collections import defaultdict

dataset = sys.argv[1]

### Paths
# Location of raw dataset (HDF5 format)
dataset_folder = "./datasets"
# Output folder
output_folder = "workflow_output"
dataset_output_folder = os.path.join(output_folder, dataset)
# Param file
param_file = os.path.join(output_folder, dataset, dataset + '.params.txt')

### Parameters
with open(param_file) as input:
    params = json.load(input)

###

print params.keys()
print params["falconn_knn_graph_trunc_py"]
n = params["num_points"]
partition = []
with open(params["kahip_output"], "r") as input:
    for line in input:
        partition.append(line.strip())
if len(partition) != n:
    raise Exception("wrong length")
counter = defaultdict(int)
for x in partition:
    counter[x] += 1
print counter
knn_graph = np.load(params["falconn_knn_graph_trunc_py"])
print knn_graph.shape
cut_size = 0
total_p2 = 0
for i in range(knn_graph.shape[0]):
    total_p2 += counter[partition[i]]
    for j in range(knn_graph.shape[1]):
        if partition[i] != partition[knn_graph[i][j]]:
            cut_size += 1
print float(cut_size) / float(knn_graph.shape[0] * knn_graph.shape[1])
print float(total_p2) / float(knn_graph.shape[0]), float(total_p2) / (float(knn_graph.shape[0])**2)
