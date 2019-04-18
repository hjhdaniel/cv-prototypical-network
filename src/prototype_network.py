# Reference Paper: Prototypical Networks for Few shot Learning in PyTorch
# Reference Paper URL: https://arxiv.org/pdf/1703.05175v2.pdf
# Reference Paper Authors: Jake Snell, Kevin Swersky, Richard S. Zemel

# Reference Code: https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch
# Reference Code Author: Daniele E. Ciriello

import torch.nn as nn
import torch
import sys
import time

class PrototypicalNetwork(nn.Module):

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
        super(PrototypicalNetwork, self).__init__()
        self.embedder = nn.Sequential(self.convolution_block(input_channel_num, hidden_channel_num),
                                    self.convolution_block(hidden_channel_num, hidden_channel_num),
                                    self.convolution_block(hidden_channel_num, hidden_channel_num),
                                    self.convolution_block(hidden_channel_num, output_channel_num))
        # FIXME Try out other dimensions
        self.fc = nn.Linear(64, 64)
        
    def convolution_block(self, input_channel_num, output_channel_num):
        return nn.Sequential(nn.Conv2d(input_channel_num, output_channel_num, 3, padding=1),
                             nn.BatchNorm2d(output_channel_num),
                             nn.ReLU(),
                             nn.MaxPool2d(2))
    
    def forward(self, input):
        input = self.embedder(input)
        # Convert 4d matrix to 2d matrix
        input = input.view(input.shape[0], -1)
        input = self.fc(input)
        return input


"""
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,
    padding=0, dilation=1, groups=1, bias=True)
"""
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        #self.fc = nn.Linear(4, 2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3,
            stride=1, padding=0, dilation=1, groups=1, bias=True)

    def forward(self, input):
        return self.conv1(input)

    

if __name__ == "__main__":

    model = PrototypicalNetwork()
    input = torch.rand((600, 1, 28, 28))
    print(input.shape)
    output = model(input)
    print(output.shape)

    sys.exit()
    model = ConvNet()
    no_channels = 1
    input = torch.tensor([
        [1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10, 11, 12],
        [13, 14, 15, 16, 17, 18],
        [19, 20, 21, 22, 23, 24]
    ], dtype=torch.float).view(1, 1, 4, 6).repeat(5, no_channels, 1, 1)

    #print(input.shape)
    output = model(input)
    print(output)