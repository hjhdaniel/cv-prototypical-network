# Reference Paper: Prototypical Networks for Few shot Learning in PyTorch
# Reference Paper URL: https://arxiv.org/pdf/1703.05175v2.pdf
# Reference Paper Authors: Jake Snell, Kevin Swersky, Richard S. Zemel

# Reference Code: https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch
# Reference Code Author: Daniele E. Ciriello

import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F

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
        print(input.shape)
        return input.view(input.size(0), -1)


class PrototypicalResnet(nn.Module):

    '''
    This network uses Resnet18 as embedding function.
    '''
    
    def __init__(self, output_size=1600):
        super(PrototypicalResnet, self).__init__()
        resnet = models.resnet18(pretrained=True)
        in_features = resnet.fc.in_features
        # Change the output size of the last layer of resnet (hyperparameter)
        # Original paper uses 1600
        resnet.fc = nn.Linear(in_features, output_size)
        self.embedder = resnet
    
    def forward(self, input):
        return self.embedder(input)

class CustomConvNet(nn.Module):
    """
    A custom convolutional net with 2 fc layers of top of it.
    Takes 3x84x84 images as input.
    Test acc with this model: ~69%
    """
    def __init__(self):
        super(CustomConvNet, self).__init__()
        self.embedder = nn.Sequential(self.convolution_block(in_channels=3, out_channels=20),
                                      self.convolution_block(in_channels=20, out_channels=60),
                                      self.convolution_block(in_channels=60, out_channels=120),
                                      self.convolution_block(in_channels=120, out_channels=240),
                                      self.convolution_block(in_channels=240, out_channels=240))
        self.fc1 = nn.Linear(960, 960)
        self.fc2 = nn.Linear(960, 500)
        self.dropout = nn.Dropout2d(p=0.3)

    def convolution_block(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(),
                             nn.MaxPool2d(2))

    def forward(self, x):
        x = self.embedder(x)
        # 240x2x2
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(self.dropout(x)))
        x = F.relu(self.fc2(self.dropout(x)))
        return x

class CustomConvNet2(nn.Module):
    """
    Test acc: 0.4101
    """
    def __init__(self, in_channels=3, hidden_channels=64, out_channels=64):
        super(CustomConvNet2, self).__init__()        
        self.embedder = nn.Sequential(self.convolution_block(in_channels, 64),
                                     self.convolution_block(64, 64),
                                     self.convolution_block(64, 64),
                                     self.convolution_block(64, 128),
                                     self.convolution_block(128, 128),
                                     self.convolution_block(128, 256))
        self.fc1 = nn.Linear(256, 256)
        
    def convolution_block(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(),
                             nn.MaxPool2d(2))
    
    def forward(self, x):
        x = self.embedder(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        return x

class FCNet(nn.Module):
    """
    Conventional net containing only fc layers.
    This is only for testing. Probably very impractical and slow to train.
    """
    def __init__(self):
        super(FCNet, self).__init__()
        # Input size of the first layer: 3*84*84=21168
        self.fc1 = nn.Linear(21168, 10000)
        self.fc2 = nn.Linear(10000, 5000)
        self.fc3 = nn.Linear(5000, 1500)
        self.dropout = nn.Dropout2d(p=0.33)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(self.dropout(x)))
        x = F.relu(self.fc2(self.dropout(x)))
        x = F.relu(self.fc3(self.dropout(x)))
        return x

    



if __name__ == "__main__":
    model = CustomConvNet2()
    input = torch.rand(20, 3, 84, 84)
    output = model(input)
    print(output.shape)