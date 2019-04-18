# Reference Paper: Prototypical Networks for Few shot Learning in PyTorch
# Reference Paper URL: https://arxiv.org/pdf/1703.05175v2.pdf
# Reference Paper Authors: Jake Snell, Kevin Swersky, Richard S. Zemel

# Reference Code: https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch
# Reference Code Author: Daniele E. Ciriello

import torch.nn as nn
import torch
import sys
import time
import torchvision.models as models

class PrototypicalNetwork(nn.Module):

    '''
    This network uses Resnet18 as embedding function.
    '''
    
    def __init__(self, input_channel_num=1, hidden_channel_num=64, output_channel_num=64):
        super(PrototypicalNetwork, self).__init__()
        resnet = models.resnet18(pretrained=True)
        in_features = resnet.fc.in_features
        # Change the output size of the last layer of resnet
        # Original paper uses 1600
        resnet.fc = nn.Linear(in_features, 1600)
        self.embedder = resnet
    
    def forward(self, input):
        return self.embedder(input)
    

if __name__ == "__main__":

    model = PrototypicalNetwork(input_channel_num=3)
    #model = models.resnet18(pretrained=True)
    input = torch.rand((10, 3, 224, 224))
    output = model(input)
    print(output.shape)