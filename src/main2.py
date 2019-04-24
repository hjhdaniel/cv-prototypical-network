# Reference Paper: Prototypical Networks for Few shot Learning in PyTorch
# Reference Paper URL: https://arxiv.org/pdf/1703.05175v2.pdf
# Reference Paper Authors: Jake Snell, Kevin Swersky, Richard S. Zemel

# Reference Code: https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch
# Reference Code Author: Daniele E. Ciriello

import os, torch
import numpy as np
from tqdm import tqdm

from prototype_network import PrototypicalNetwork
from gaussian_prototype_network import GaussianPrototypicalNetwork
from dataloader import DataLoader
from loss_function import prototypical_loss as loss_func, gaussian_prototypical_loss as gaussian_loss_func
from arg_parser import get_parser
from utils import *


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
    # print("Support idx: {}".format(support_idxs))
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

    prototypes = generate_prototype(model_output, y, n_support, classes)
    query_idxs = torch.stack(list(map(lambda c: y.eq(c).nonzero()[n_support:], classes))).view(-1)
    query_samples = model_output[query_idxs]
    return n_classes, n_query, prototypes, query_samples

def initialise_loss_samples_gaussian(model_output, y, n_support):
    """"""
    # Assume diagonal mode for now
    # x_embedding_points, sigma = torch.split(model_output, int(model_output.size(1)/2), dim=1)
    classes = torch.unique(y)
    n_classes = len(classes)
    n_query = y.eq(classes[0].item()).sum().item() - n_support

    gaussian_prototypes, support_inv_sigmas = generate_gaussian_prototype(model_output, y, n_support, classes)
    query_idxs = torch.stack(list(map(lambda c: y.eq(c).nonzero()[n_query:], classes))).view(-1)
    query_samples = model_output[query_idxs]
    return n_classes, n_query, query_samples, gaussian_prototypes, support_inv_sigmas

def generate_gaussian_prototype(model_output, y, n_support, classes):
    support_idxs = list(map(lambda c: y.eq(c).nonzero()[:n_support].squeeze(1), classes))
    x_embedding_points, sigmas = torch.split(model_output, int(model_output.size(1)/2), dim=1)
    sigmas = torch.nn.functional.softplus(sigmas) + 1
    # embeddings of support points per class
    support_samples = [x_embedding_points[idx_list] for idx_list in support_idxs]
    # sigmas per class
    support_sigmas = [sigmas[idx_list] for idx_list in support_idxs]
    
    prototypes = []
    sigmas_inverse = []
    for idx, support_sample in enumerate(support_samples):
        summation_numerator = torch.sum(support_sample * support_sigmas[idx], dim=0)
        summation_denominator = torch.sum(support_sigmas[idx], dim=0)
        prototypes.append(summation_numerator / summation_denominator)
        sigmas_inverse.append(summation_denominator)

    return torch.stack(prototypes), torch.stack(sigmas_inverse)



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
            n_classes, n_query, query_samples, prototypes, support_inv_sigmas = initialise_loss_samples_gaussian(model_output, y, n_support)

            loss, acc = loss_function(device, n_classes, n_query, prototypes, query_samples, support_inv_sigmas)
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
    # model = PrototypicalNetwork().to(device)
    model = GaussianPrototypicalNetwork().to(device)

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
                                                                           loss_function=gaussian_loss_func)

    # test the best model from training
    test(device=device, arg_settings=arg_settings, testing_dataloader=testing_dataloader, 
         model=model ,loss_function=loss_func)

if __name__=='__main__':
    main()