import torch
import numpy as np
import torch.nn.functional as F
import torch.nn
import math
from torch.autograd import Variable
import scipy
from scipy import io
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

use_cuda = torch.cuda.is_available()


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features,bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        input = torch.squeeze(input)
        adj = torch.squeeze(adj)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class DeepEZ(torch.nn.Module):
    def __init__(self, example, num_classes=10):
        super(DeepEZ, self).__init__()
        self.gc1 = GraphConvolution(246,120)
        self.gc2 = GraphConvolution(120,50)
        self.fc_class = torch.nn.Linear(50,2,bias=False)
        self.fc1 = torch.nn.Linear(246,60)
        self.fc2 = torch.nn.Linear(60,1)

    def forward(self, x, adj):
        out = F.leaky_relu(self.gc1(x, adj),negative_slope=0.1)
        out = F.leaky_relu(self.gc2(out, adj), negative_slope=0.1)
        out = F.leaky_relu(self.fc_class(out), negative_slope=0.1)

        ann_out = F.leaky_relu(self.fc1(out.view(out.size(1),out.size(0))),negative_slope=0.1)
        ann_out = F.leaky_relu(self.fc2(ann_out),negative_slope=0.1)

        out = torch.add(out,ann_out.view(ann_out.size(1),ann_out.size(0)))

        return out, ann_out


import torch.utils.data.dataset

total_pred = []
total_acc = []
total_auc = []
total_sens = []
total_spec = []
total_precision = []
lr = 0.005
nbepochs = 200
BATCH_SIZE = 1
class_0 = 0.29
class_1 = 1.52
Alpha = 0.017


for test_range in range(14):
    test_index = test_range + 1
    trainset = data_train(index=test_index, fold=14)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
    testset = data_test(index=test_index)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

    net = DeepEZ(trainset.X)
    if use_cuda:
        net = net.cuda(0)
        net = torch.nn.DataParallel(net, device_ids=[0])

    momentum = 0.9
    wd = 0.00005  ## Decay for L2 regularization


    def init_weights_he(m):

        if type(m) == torch.nn.Linear:
            fan_in = net.dense1.in_features
            he_lim = np.sqrt(6) / fan_in
            m.weight.data.uniform_(-he_lim, he_lim)
            print(m.weight)


    class_weight = torch.FloatTensor([class_0, class_1])
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=wd)
    optimizer2 = torch.optim.Adam(net.parameters(), lr=lr)

    def train(epoch, alpha=1.0,idx=1.0):
        net.train()

        for batch_idx, (X,A,Y) in enumerate(trainloader):
            if use_cuda:
                X,A, Y = X.cuda(),A.cuda(), Y.cuda()
            optimizer.zero_grad()

            X, A, Y = Variable(X),Variable(A), Variable(Y)
            out, ann_out = net(X,A)

            Y = Y.view(Y.size(0) * Y.size(1), 1)
            Y = np.squeeze(Y)
            Y = Variable(Y)

            loss_backprop = torch.zeros((1, 1), requires_grad=True)
            loss = criterion((out), Y)

            loss_epilepsy = torch.zeros((1, 1), requires_grad=True)
            for i in range(246):
                if Y[i] == 1:
                    loss_epilepsy = torch.add(-out[i,1], loss_epilepsy)

            loss = loss + alpha * torch.mean(loss_epilepsy)
            loss.backward()
            optimizer2.step()
        return loss_backprop

    def test(alpha=1.0):
        net.eval()
        total_out = []
        for batch_idx, (X,A, Y) in enumerate(testloader):
            if use_cuda:
                X, A, Y = X.cuda(), A.cuda(), Y.cuda()
            with torch.no_grad():
                if use_cuda:
                    X,A, Y = X.cuda(), A.cuda(), Y.cuda()
                optimizer.zero_grad()
                X, A, Y = Variable(X), Variable(A), Variable(Y)
                out, ann_out = net(X, A)
                out = out.cpu()
                out = out.data.numpy()
                total_out.append(out)
        return total_out, ann_out

    cont = scipy.io.loadmat('BNA_cont.mat', mdict=None)
    cont = cont['cont']

    for epoch in range(nbepochs):
        train(epoch,alpha=Alpha,idx=cont)

    out, ann_out = test()

