# Reference Paper: Prototypical Networks for Few shot Learning in PyTorch
# Reference Paper URL: https://arxiv.org/pdf/1703.05175v2.pdf
# Reference Paper Authors: Jake Snell, Kevin Swersky, Richard S. Zemel

# Reference Code: https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch
# Reference Code Author: Daniele E. Ciriello

import torch.nn as nn

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
                                     self.convolution_block(hidden_channel_num, output_channel_num),)
        
    def convolution_block(self, input_channel_num, output_channel_num):
        return nn.Sequential(nn.Conv2d(input_channel_num, output_channel_num, 3, padding=1),
                             nn.BatchNorm2d(output_channel_num),
                             nn.ReLU(),
                             nn.MaxPool2d(2))
    
    def forward(self, input):
        input = self.embedder(input)
        return input.view(input.size(0), -1)