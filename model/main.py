import torch
import torch.nn as nn
import argparse
from torch.utils.data import Dataset

import sys

'''
Block of net
'''
def net_block(n_in, n_out):
    
    block = nn.Sequential(nn.Linear(n_in, n_out),
                          nn.BatchNorm1d(n_out),
                          nn.ReLU())
    return block
    
class Model(nn.Module):
    def __init__(self, n_input, n_hidden, num_class, opt, toplevel=False):
        super(Model, self).__init__()
        self.opt = opt
        self.toplevel = toplevel
        
        self.block1 = net_block(n_input, n_hidden)
        self.dropout = nn.Dropout(p=0.1)
        
        if (opt.glove or opt.sift or opt.prefix10m):
            #if include skip connection:
            #self.block_mid = net_block(n_hidden + n_input, n_hidden)            
            self.block_mid = net_block(n_hidden, n_hidden)            
        if toplevel:
            self.block2 = net_block(n_hidden, n_hidden)
        
        self.fc1 = nn.Linear(n_hidden, num_class)

        self.softmax = nn.Softmax(dim=-1)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.01)

    def forward(self, x):

        y = self.block1(x)
        #y = self.dropout(x1)
        
        if self.opt.glove or self.opt.sift or self.opt.prefix10m:
            #if include skip connection:
            #y = self.block_mid(torch.cat([x, y], dim=1))
            y = self.block_mid(y)
            
        if self.toplevel:            
            y = self.block2(y)
            y = self.dropout(y)
            
        out = self.fc1(y)
        out = self.softmax(out)
        return out

def get_dataset(data, shuffle, param_batch_size):
    X, y = data
    dset = torch.utils.data.TensorDataset(X.float(), y)
    loader = torch.utils.data.DataLoader(dataset=dset, batch_size=param_batch_size,
                                         shuffle=shuffle)
    return loader


def write_results(result, output):
    result = (-result.detach().numpy()).argsort(axis=1)
    for i in range(result.shape[0]):
        output.write(" ".join([str(x) for x in result[i]]) + "\n")

        
def run(param_feat, param_lr, param_batch_size):
    print("RUNNING WITH: features="+str(param_feat)+"; lr="+str(param_lr)+"; batch_size="+str(param_batch_size))
    input_dim = 100
    # read data
    X, y = torch.load('./data/parts64/data.path')
    import numpy as np
    dataset = np.load('./data/parts64/dataset.npy')
    queries = np.load('./data/parts64/queries.npy')
    n_data = X.size(0)
    split = int(n_data * 0.95)
    trainloader = get_dataset((X[:split], y[:split]), shuffle=True, param_batch_size=param_batch_size)
    valloader = get_dataset((X[split:], y[split:]), shuffle=False, param_batch_size=param_batch_size)

    # build model
    m = Model
    model = m(input_dim=input_dim, feat_dim=param_feat, num_class=64, args=None).cuda()

    # criterion
    crit = nn.CrossEntropyLoss().cuda()

    # optimizer
    # optimizer = torch.optim.RMSprop(model.parameters(), args.lr)
    lr = param_lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=10**(-4))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30, 35, 38, 39], gamma=0.1)

    # start training!
    losses = []
    iterations = 40
    for ep in range(1, iterations + 1):
        print("==="+str(ep)+"===")
        loss_sum = 0.
        train_acc_tot = 0
        train_n_tot = 0
        scheduler.step()
        for i, (X, y) in enumerate(trainloader):
            y_pred = model(X.cuda())
            loss = crit(y_pred, y.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

            train_acc_tot += (y_pred.argmax(dim=1).cpu() == y).sum().item()
            train_n_tot += X.size(0)

        print("loss:", loss_sum)
        print("train acc:", train_acc_tot*1. / train_n_tot)
        losses.append(loss_sum / len(trainloader))
        acc_tot = 0
        n_tot = 0.
        for i, (X, y) in enumerate(valloader):
            y_pred = model(X.cuda())
            acc_tot += (y_pred.argmax(dim=1).cpu() == y).sum().item()
            n_tot += X.size(0)

        print("val acc:", acc_tot / n_tot)

    print("Doing inference and writing result files...")
    # inference on data
    batch_size = 10000
    param_str = "_".join(sys.argv[1:])
    with open("./data/parts64/data_prediction"+param_str+".txt","w") as output:
        for b in range(0, n_data, batch_size):
            data_batch_results = model(torch.from_numpy(dataset[b:b+batch_size]).float().cuda()).cpu()
            write_results(data_batch_results, output)

    # inference on queries
    query_results = model(torch.from_numpy(queries).float().cuda()).cpu()
    with open("./data/parts64/queries_prediction"+param_str+".txt","w") as output:
        write_results(query_results, output)


if __name__ == "__main__":
    run(int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]))
