import sys
import numpy as np

if sys.version.startswith("2"):
    a = np.load(open('./dataset.npy'))
    # this may take ~50s, be patient!
    np.savetxt("try.csv", a, delimiter=",")
else:
    import torch

    label = np.loadtxt(open("labels.txt"))
    label = torch.from_numpy(label)
    feat = np.loadtxt(open("try.csv"), delimiter=",")
    feat = torch.from_numpy(feat)

    torch.save((label, feat), "data.path")
