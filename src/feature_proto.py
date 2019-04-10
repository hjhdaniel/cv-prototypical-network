import torch.nn as nn

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
        self.conv2 = self.convolution_block(input_channel_num, hidden_channel_num)
        self.conv3 = self.convolution_block(input_channel_num, hidden_channel_num)
        self.conv4 = self.convolution_block(input_channel_num, hidden_channel_num)
        
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
        return conv1_out, conv2_out, conv3_out, conv4_out, output