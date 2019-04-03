# Reference Paper: Prototypical Networks for Few shot Learning in PyTorch
# Reference Paper URL: https://arxiv.org/pdf/1703.05175v2.pdf
# Reference Paper Authors: Jake Snell, Kevin Swersky, Richard S. Zemel

# Reference Code: https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch
# Reference Code Author: Daniele E. Ciriello

import os, torch
import numpy as np

from prototype_network import PrototypicalNetwork
from dataloader import DataLoader
from loss_function import prototypical_loss as loss_func
from arg_parser import get_parser
from train import train
from test import test

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
    model = PrototypicalNetwork().to(device)

    # initialise optimizer: Adaptive Moment Estimation (Adam)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=arg_settings.learning_rate)

    # initialise learning rate scheduler
    torch.optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=arg_settings.lr_scheduler_gamma,
                                    step_size=arg_settings.lr_scheduler_step)

    # train model, obtain results from training and save the best model
    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = train(arg_settings=arg_settings,
                                                                           training_dataloader=training_dataloader,
                                                                           validation_dataloader=validation_dataloader,
                                                                           model=model,
                                                                           optimizer=optimizer,
                                                                           lr_scheduler=lr_scheduler,
                                                                           loss_function=loss_func)

    # test the best model from training
    test(arg_settings=arg_settings, testing_dataloader=testing_dataloader, 
         model=model ,loss_function=loss_func)

if __name__=='__main__':
    main()