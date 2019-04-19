import os, torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules import Module
import numpy as np
from tqdm import tqdm

from dataloader import DataLoader
from arg_parser import get_parser
from utils import *


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


def prototypical_loss(device, n_classes, n_query, feature_prototypes, feature_query_samples):
    # TODO: Change to for each query set of each feature, compare with respective prototype feature

    feature_dists = []
    for query_samples, prototypes in zip(feature_query_samples, feature_prototypes):
        dist = euclidean_dist(query_samples, prototypes)
        feature_dists.append(dist)
    mean_dist = sum(feature_dists) / len(feature_dists)

    log_p_y = F.log_softmax(-mean_dist, dim=1).view(n_classes, n_query, -1)
    log_p_y = log_p_y.to(device)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()
    target_inds = target_inds.to(device)
    
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    return loss_val, acc_val


class FeaturePrototypicalNetwork(nn.Module):
    '''
    This network non-linearly maps the input into an embedding space.
    It consists of 5 convolution blocks
    Each convolution block consists of:
        -3X3 convolution
        -Batch normalization
        -ReLU non-linearity
        -2X2 maxpooling
    '''
    def __init__(self, input_channel_num=1, hidden_channel_num=64, output_channel_num=64):
        super(FeaturePrototypicalNetwork, self).__init__()        
        self.conv1 = self.convolution_block(input_channel_num, hidden_channel_num)
        self.conv2 = self.convolution_block(hidden_channel_num, hidden_channel_num)
        self.conv3 = self.convolution_block(hidden_channel_num, hidden_channel_num)
        self.conv4 = self.convolution_block(hidden_channel_num, output_channel_num)
        
    def convolution_block(self, input_channel_num, output_channel_num):
        return nn.Sequential(nn.Conv2d(input_channel_num, output_channel_num, 3, padding=1),
                             nn.BatchNorm2d(output_channel_num),
                             nn.ReLU(),
                             nn.MaxPool2d(2))
    
    def forward(self, input):
        conv1_out = self.conv1(input)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        output = conv4_out.view(conv4_out.size(0), -1)
        conv1_out = torch.mean(conv1_out, (3, 2)) # sum over feature space
        conv2_out = torch.mean(conv2_out, (3, 2))
        conv3_out = torch.mean(conv3_out, (3, 2))
        return conv1_out, conv2_out, conv3_out, output


def generate_prototype(model_output, y, n_support, classes):
    """
    Generate prototype for each class
    Args:
        model_output: embeddings for all classes in batch
        y: class labels in batch
        n_support: length of support set
        classes: unique classes labels
    Return:
        prototypes: tensor of mean embedding of each class, shape(60, 64)
    """
    support_idxs = list(map(lambda c: y.eq(c).nonzero()[:n_support].squeeze(1), classes))
    prototypes = torch.stack([model_output[idx_list].mean(0) for idx_list in support_idxs])
    return prototypes


def initialise_loss_samples(model_output, y, n_support):
    """
    Args:
        model_output: embeddings for all classes in batch
        y: class labels in batch
        n_support: length of support set
    Return:
        n_classes: number of unique classes
        n_query: length of query set
        prototypes: tensor of mean embedding of each class, shape(60, 64)
        query_samples: embeddings to use for query set
    """
    classes = torch.unique(y)
    n_classes = len(classes)
    n_query = y.eq(classes[0].item()).sum().item() - n_support

    feature_prototypes = []
    feature_query_samples = []

    for feature in model_output:
        prototypes = generate_prototype(feature, y, n_support, classes)
        query_idxs = torch.stack(list(map(lambda c: y.eq(c).nonzero()[n_support:], classes))).view(-1)
        query_samples = feature[query_idxs]
        feature_prototypes.append(prototypes)
        feature_query_samples.append(query_samples)

    return n_classes, n_query, feature_prototypes, feature_query_samples


def train(device, arg_settings, training_dataloader, model, optimizer, 
          lr_scheduler, loss_function, validation_dataloader=None):

    if validation_dataloader is None:
        best_state = None
    
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0

    best_model_path = os.path.join(arg_settings.experiment_root, 'best_model.pth')
    last_model_path = os.path.join(arg_settings.experiment_root, 'last_model.pth')

    for epoch in range(arg_settings.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        
        # Training
        tr_iter = iter(training_dataloader)
        model.train()
        for batch in tqdm(tr_iter):
            optimizer.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x) # Embedding

            # Create prototype, separated from loss fuction
            n_support = arg_settings.num_support_tr
            n_classes, n_query, feature_prototypes, feature_query_samples = initialise_loss_samples(model_output, y, n_support)

            loss, acc = loss_function(device, n_classes, n_query, feature_prototypes, feature_query_samples)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())

        avg_loss = np.mean(train_loss[-arg_settings.iterations:])
        avg_acc = np.mean(train_acc[-arg_settings.iterations:])
        print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        lr_scheduler.step()
        if validation_dataloader is None:
            continue
        
        # Validation
        val_iter = iter(validation_dataloader)
        model.eval()
        for batch in val_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x) # Embedding

            # Create prototype, separated from loss fuction
            n_support = arg_settings.num_support_val
            n_classes, n_query, prototypes, query_samples = initialise_loss_samples(model_output, y, n_support)

            loss, acc = loss_function(device, n_classes, n_query, prototypes, query_samples)
            val_loss.append(loss.item())
            val_acc.append(acc.item())

        avg_loss = np.mean(val_loss[-arg_settings.iterations:])
        avg_acc = np.mean(val_acc[-arg_settings.iterations:])
        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
            best_acc)
        print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
            avg_loss, avg_acc, postfix))
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()

    torch.save(model.state_dict(), last_model_path)

    for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
        save_list_to_file(os.path.join(arg_settings.experiment_root,
                                       name + '.txt'), locals()[name])

    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc


def test(device, arg_settings, testing_dataloader, model, loss_function):
    avg_acc = list()
    for epoch in range(10):
        test_iter = iter(testing_dataloader)
        for batch in test_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x) # Embedding

            # Create prototype, separated from loss fuction
            n_support = arg_settings.num_support_tr
            n_classes, n_query, prototypes, query_samples = initialise_loss_samples(model_output, y, n_support)

            loss, acc = loss_function(device, n_classes, n_query, prototypes, query_samples)
            avg_acc.append(acc.item())

    avg_acc = np.mean(avg_acc)
    print('Test Acc: {}'.format(avg_acc))
    return avg_acc


def main():
    # initialise parser for arguments
    arg_settings = get_parser().parse_args()

    # get directory root for saving models, losses and accuracies
    if not os.path.exists(arg_settings.experiment_root):
        os.makedirs(arg_settings.experiment_root)
        
    # check if GPU is available
    if torch.cuda.is_available() and not arg_settings.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # set seed so results can be reproduced
    torch.cuda.cudnn_enabled = False
    np.random.seed(arg_settings.seed)
    torch.manual_seed(arg_settings.seed)
    torch.cuda.manual_seed(arg_settings.seed)

    # load training, testing and validation datasets
    training_dataloader = DataLoader('train', arg_settings).data_loader
    testing_dataloader = DataLoader('test', arg_settings).data_loader
    validation_dataloader = DataLoader('val', arg_settings).data_loader

    # initialise prototypical network model (utilise GPU if available)
    device = 'cuda:0' if torch.cuda.is_available() and arg_settings.cuda else 'cpu'
    model = FeaturePrototypicalNetwork().to(device)

    # initialise optimizer: Adaptive Moment Estimation (Adam)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=arg_settings.learning_rate)

    # initialise learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=arg_settings.lr_scheduler_gamma,
                                    step_size=arg_settings.lr_scheduler_step)

    # train model, obtain results from training and save the best model
    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = train(device=device, 
                                                                           arg_settings=arg_settings,
                                                                           training_dataloader=training_dataloader,
                                                                           validation_dataloader=validation_dataloader,
                                                                           model=model,
                                                                           optimizer=optimizer,
                                                                           lr_scheduler=lr_scheduler,
                                                                           loss_function=prototypical_loss)

    # test the best model from training
    test(device=device, arg_settings=arg_settings, testing_dataloader=testing_dataloader, 
         model=model ,loss_function=prototypical_loss)


if __name__=='__main__':
    main()