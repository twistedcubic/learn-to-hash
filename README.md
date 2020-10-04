# Learning Space Partitions for Nearest Neighbor Search #

This is the code for our paper [**Learning Space Partitions for Nearest Neighbor Search**](https://arxiv.org/abs/1901.08544).

Yihe Dong, Piotr Indyk, Ilya Razenshteyn, Tal Wagner.
_________________

The code is structured around a few centralized scripts, with additional versatile components that implement a common interface, which can be swapped to run different learning methods. This modular structure allows easy extensibility of our framework to new learning methods.

# Code layout #

[kahip/kmkahip.py](kahip/kmkahip.py): main backbone for creating knn graphs from dataset, recursively partitioning dataset using [KaHIP](https://github.com/KaHIP/KaHIP) in parallel, and learning tree of neural networks in tandem with building partitions tree.

[workflow_learn_kmeans](workflow_learn_kmeans.py): Main pipeline for running unsupervised learning methods: k-means, PCA, ITQ, random projection.

There are numerous additional scripts that provide utilities for training, model construction, partitioning, knn graph creation, data processing, etc.

# Demo #

For any given configuration as specified in utils.py, one can run any script with a __main__ function. We provide two demo scripts as simple examples:

[demo_km.py](demo_km.py): unsupervised learning, can use methods including k-means, PCA tree, ITQ, and random projection.

[demo_train.py](demo_train.py): supervised learning using KaHIP partitions, training can be either neural networks or logistic regression.

To change the various configurations, such as adjusting between different learning methods or datasets, one can either modify the corresponding options under the parse_args function, or pass them in through the command line.

# Sample Data #
The various algorithms were designed to have a unified requirement for input data, specifically, they require `dataset`, `queryset`, and `neighbors`, where the `dataset` and `queryset` are `n x d` arrays, where `n` denotes the number of datapoints or queries, respectively, and `d` denotes the datapoint dimension. `neighbors` is an `n x k` array of indices of the `k` nearest neighbors for each query point amongst all datapoints, where `n` denotes the number of queries, and `k` denotes the number of nearest neighbors being sought.

As an example, we provide a [queryset array for GloVe](data/glove_queries.npy).

The dataset array takes the exact same format, and is not contained in this repo due to space limitations. But which can be [downloaded here](https://github.com/erikbern/ann-benchmarks).

The neighbors array can be easily obtained a number of ways, for instance by using the [`utils`](utils.py) function:
```
utils.dist_rank(queryset, k=opt.k, data_y=dataset, largest=False)
```

# Prerequisites #

* PyTorch 0.4 or above.
* [KaHIP](https://github.com/KaHIP/KaHIP)
* scikit-learn (version insensitive)
* numpy (version insensitive)

# Directory configuration #

To point to the various resources needed at runtime, the file [config](config) in the base directory needs to contain the following absolute paths:

kahip_dir: directory to KaHIP installation.

data_dir: data directory, containing data such as knn graphs, and to contain various data produced at runtime.

glove_dir: directory containing GloVe partitions, specifically, subdirectories named "partition_256_strong" containing output of KaHIP in a file named "partition.txt".

sift_dir: directory containing SIFT partitions, analogous to above. E.g. subdirectories named "partition_16_strong" containing output of KaHIP in a file named "partition.txt".

### Reference

If you find our paper and repo useful, please cite as:

```
@article{neural_lsh2019,
  title={Learning Space Partitions for Nearest Neighbor Search},
  author={Dong, Yihe and Indyk, Piotr and Razenshteyn, Ilya and Wagner, Tal},
  booktitle={International Conference on Learning Representations},
  year={2020}
}
```

