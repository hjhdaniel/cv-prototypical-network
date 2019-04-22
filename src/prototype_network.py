# Reference Paper: Prototypical Networks for Few shot Learning in PyTorch
# Reference Paper URL: https://arxiv.org/pdf/1703.05175v2.pdf
# Reference Paper Authors: Jake Snell, Kevin Swersky, Richard S. Zemel

# Reference Code: https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch
# Reference Code Author: Daniele E. Ciriello

import torch.nn as nn
import torch
import torchvision.models as models

class PrototypicalNetwork(nn.Module):

    '''
    The original version of the prototypical network.

    This network non-linearly maps the input into an embedding space.
    It consists of 5 convolution blocks
    Each convolution block consists of:
        -3X3 convolution
        -Batch normalization
        -ReLU non-linearity
        -2X2 maxpooling
    '''
    
    def __init__(self, input_channel_num=1, hidden_channel_num=64, output_channel_num=64):
        super(PrototypicalNetwork, self).__init__()        
        self.embedder = nn.Sequential(self.convolution_block(input_channel_num, hidden_channel_num),
                                     self.convolution_block(hidden_channel_num, hidden_channel_num),
                                     self.convolution_block(hidden_channel_num, hidden_channel_num),
                                     self.convolution_block(hidden_channel_num, output_channel_num),)
        
    def convolution_block(self, input_channel_num, output_channel_num):
        return nn.Sequential(nn.Conv2d(input_channel_num, output_channel_num, 3, padding=1),
                             nn.BatchNorm2d(output_channel_num),
                             nn.ReLU(),
                             nn.MaxPool2d(2))
    
    def forward(self, input):
        input = self.embedder(input)
        return input.view(input.size(0), -1)


class PrototypicalResnet(nn.Module):

    '''
    This network uses Resnet18 as embedding function.
    '''
    
    def __init__(self, input_channel_num=3, output_size=1600):
        super(PrototypicalResnet, self).__init__()
        resnet = models.resnet18(pretrained=True)
        in_features = resnet.fc.in_features
        # Change the output size of the last layer of resnet (hyperparameter)
        # Original paper uses 1600
        resnet.fc = nn.Linear(in_features, output_size)
        self.embedder = resnet
    
    def forward(self, input):
        return self.embedder(input)


if __name__ == "__main__":
    model = models.resnet18(pretrained=True)
    input = torch.rand(20, 3, 224, 224)
    output = model(input)
    print(output.shape)