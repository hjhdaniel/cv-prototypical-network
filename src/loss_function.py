# Reference Paper: Prototypical Networks for Few shot Learning in PyTorch
# Reference Paper URL: https://arxiv.org/pdf/1703.05175v2.pdf
# Reference Paper Authors: Jake Snell, Kevin Swersky, Richard S. Zemel

# Reference Code: https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch
# Reference Code Author: Daniele E. Ciriello

import torch
from torch.nn import functional as F
from torch.nn.modules import Module

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
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def mahalanobis_dist(x, proto_mean, proto_cov_mat):
    '''
    Compute mahalanobis distance between query inputs and prototype gaussian distributions
    :param x: query inputs
    :param proto_mean: mean of prototype gaussian distributions
    :param proto_cov_mat: covariance matrices of prototype gaussian distributions
    '''

    # maha_dists = torch.zeros([x.size()[0], proto_mean.size()[0]], dtype=torch.float64)
    # for input_idx in range(x.size()[0]):
    #     for proto_idx in range(proto_mean.size()[0]):
    #         x_subtract_mean = x[input_idx] - proto_mean[proto_idx]
    #         cov_mat_inv = torch.pinverse(proto_cov_mat[proto_idx])
    #         x_subtract_mean = x_subtract_mean.view(1, -1)
    #         maha_dists[input_idx][proto_idx] = torch.sqrt(torch.abs(torch.mm(torch.mm(x_subtract_mean, cov_mat_inv), torch.t(x_subtract_mean))))

    # initialize size parameters
    n_inputs = x.size(0)
    n_classes = proto_mean.size(0)
    n_features = x.size(1)
    if n_features!=proto_mean.size(1):
        raise Exception

    # obtain matrix of x subtract mean
    x = x.unsqueeze(1).expand(n_inputs, n_classes, n_features)
    proto_mean = proto_mean.unsqueeze(0).expand(n_inputs, n_classes, n_features)
    x_subtract_mean = x - proto_mean

    # get inverse of covariance matrices
    proto_cov_mat_inv = torch.zeros(proto_cov_mat.size())
    for proto_idx in range(n_classes):
        proto_cov_mat_inv[proto_idx] = torch.pinverse(proto_cov_mat[proto_idx])

    # reshape the matrices for batch matrix multiplication operations
    x_subtract_mean = x_subtract_mean.unsqueeze(2).expand(n_inputs, n_classes, 1, n_features)
    x_subtract_mean_t = x_subtract_mean.transpose(2, 3)
    proto_cov_mat_inv = proto_cov_mat_inv.unsqueeze(0).expand(n_inputs, n_classes, n_features, n_features)

    # matrix multiplication for mahalanobis distance
    maha_dists = x_subtract_mean @ proto_cov_mat_inv @ x_subtract_mean_t

    # get absolute values (distances) and square-root of mahalanobis distance. Reshape to (n_inputs, n_classes)
    maha_dists = torch.sqrt(torch.abs(maha_dists.view(n_inputs, n_classes)))

    return maha_dists

def gaussian_prototypical_loss(device, n_classes, n_query, proto_mean, proto_cov_mat, query_samples):
    '''
    Get average loss and accuracy of predicting query points based on gaussian distribution of support sets
    '''

    dists = mahalanobis_dist(query_samples, proto_mean, proto_cov_mat)

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

    dists = euclidean_dist(query_samples, prototypes)

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