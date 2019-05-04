import torch.nn as nn

class GaussianPrototypicalNetwork(nn.Module):
    def __init__(self, input_channel_num=1, hidden_channel_num=64, outout_channel_num=64, gaussian_mode="diagonal"):
        super(GaussianPrototypicalNetwork, self).__init__()
        if gaussian_mode == "diagonal":
            self.last_convolutional_block = self.convolution_block(hidden_channel_num, 2*hidden_channel_num)
        elif gaussian_mode == "radial":
            self.last_convolutional_block = self.convolution_block(hidden_channel_num, hidden_channel_num+1)
        else:
            print("Invalid gaussian mode specified")
            return
        self.embedder = nn.Sequential(
            self.convolution_block(input_channel_num, hidden_channel_num),
            self.convolution_block(hidden_channel_num, hidden_channel_num),
            self.convolution_block(hidden_channel_num, hidden_channel_num),
            self.last_convolutional_block,
        )
    
    def convolution_block(self, input_channel_num, output_channel_num):
        return nn.Sequential(
            nn.Conv2d(input_channel_num, output_channel_num, 3, padding=1),
            nn.BatchNorm2d(output_channel_num),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    
    def forward(self, input):
        input = self.embedder(input)
        return input.view(input.size(0), -1)