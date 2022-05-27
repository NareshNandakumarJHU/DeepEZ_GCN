import torch
import numpy as np
import torch.nn.functional as F
import torch.nn
from torch.autograd import Variable
import os.path
import scipy
from scipy import io
import sys

class data_test(torch.utils.data.Dataset):
    def __init__(self, transform=False, class_balancing=False, index=0):
        self.transform = transform

        stringX = 'x' + str(index) + '.mat'
        X_str = 'X'

        stringA = 'A_GCN.mat'
        A_str = 'A_GCN'

        stringY = 'y' + str(index) + '.mat'
        Y_str = 'y'

        x = scipy.io.loadmat(stringX, mdict=None)
        x = x[X_str]

        y = scipy.io.loadmat(stringY, mdict=None)
        y = y[Y_str]

        A = scipy.io.loadmat(stringA, mdict=None)
        A = A[A_str]

        self.X = torch.FloatTensor(np.expand_dims(x, 1).astype(np.float32))
        self.A = torch.FloatTensor(np.expand_dims(A, 1).astype(np.float32))
        self.Y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sample = [self.X[idx],self.A[idx], self.Y[idx]]
        if self.transform:
            sample[0] = self.transform(sample[0])
        return sample


class data_train(torch.utils.data.Dataset):
    def __init__(self, transform=False, class_balancing=False, index=0, fold=1):
        self.transform = transform

        folds_index = range(fold)
        folds_index = [x + 1 for x in folds_index]
        del [folds_index[index - 1]]
        count = 0
        for i in folds_index:
            stringX = 'x' + str(i) + '.mat'
            X_str = 'X'

            stringA = 'A_GCN.mat'
            A_str = 'A_GCN'

            stringY = 'y' + str(i) + '.mat'
            Y_str = 'y'

            A_partial = scipy.io.loadmat(stringA, mdict=None)
            A_partial = A_partial[A_str]

            x_partial = scipy.io.loadmat(stringX, mdict=None)
            x_partial = x_partial[X_str]

            y_partial = scipy.io.loadmat(stringY, mdict=None)
            y_partial = y_partial[Y_str]


            if count == 0:
                x = x_partial
                A = A_partial
                y = y_partial

            else:
                x = np.concatenate((x, x_partial), axis=0)
                A = np.concatenate((A,A_partial), axis=0)
                y = np.concatenate((y, y_partial), axis=0)

            count = count + 1


        self.X = torch.FloatTensor(np.expand_dims(x, 1).astype(np.float32))
        self.A = torch.FloatTensor(np.expand_dims(A, 1).astype(np.float32))
        self.Y = torch.tensor(y, dtype=torch.long)


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sample = [self.X[idx], self.A[idx], self.Y[idx]]
        if self.transform:
            sample[0] = self.transform(sample[0])
        return sample