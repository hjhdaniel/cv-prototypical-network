# Reference Paper: Prototypical Networks for Few shot Learning in PyTorch
# Reference Paper URL: https://arxiv.org/pdf/1703.05175v2.pdf
# Reference Paper Authors: Jake Snell, Kevin Swersky, Richard S. Zemel

# Reference Code: https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch
# Reference Code Author: Daniele E. Ciriello

import torch
from torch.nn import functional as F
from torch.nn.modules import Module
import numpy as np

class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    print(x.size())
    print(y.size())
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    print(torch.pow(x - y, 2).sum(2).size())
    return torch.pow(x - y, 2).sum(2)

def mahalanobis_dist(x, y):
    '''
    Compute mahalanobis distance between two tensors
    '''
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    # x = x.unsqueeze(1).expand(n, m, d)
    # y = y.unsqueeze(0).expand(n, m, d)
    # x_numpy = x.detach().cpu().numpy()
    # y_numpy = y.detach().cpu().numpy()
    dists = torch.empty((n, m))

    cov = get_cov_mat(torch.transpose(x, 0, 1))  # x.T: row - variables, col - observations
    cov_inv = torch.inverse(cov)
    for i in range(m):
        delta = x - y[i]
        dists[:, i] = torch.sqrt(torch.einsum('ij,jj,ij->i', delta, cov_inv, delta))

    return dists

def get_cov_mat(matrix):
    '''
    Get covariance matrix that describes the gaussian distribution of support inputs
    :param matrix: matrix of 1 set of support inputs with dimensions m x n
           m: number of features
           n: number of observations
    :return: covariance matrix
    '''
    matrix_mean = torch.mean(matrix, dim=1)
    x = matrix - matrix_mean[:, None]
    cov_mat = 1 / (x.size(1) - 1) * x.mm(x.t())
    return cov_mat

def manhattan_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.abs(x - y).sum(2)


def prototypical_loss(device, n_classes, n_query, prototypes, query_samples):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py
    Compute the prototypes by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the prototypes, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    '''
    dists = mahalanobis_dist(query_samples, prototypes)

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
    log_p_y = log_p_y.to(device)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()
    target_inds = target_inds.to(device)
    
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    return loss_val,  acc_val