# List of APIs
import torch
import torch.nn as nn


class DenseBlock(nn.Module):
    '''
    param ={
        'num_channels':1,
        'num_filters':64,
        'kernel_h':5,
        'kernel_w':5,
        'stride_conv':1,
        'pool':2,
        'stride_pool':2,
        'num_classes':28
    }

    '''

    def __init__(self, params):
        super(DenseBlock, self).__init__()

        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)

        conv1_out_size = int(params['num_channels']+params['num_filters'])
        conv2_out_size = int(params['num_channels']+params['num_filters']+params['num_filters'])

        self.conv1 = nn.Conv2d(in_channels=params['num_channels'], out_channels=params['num_filters'],
                              kernel_size=(params['kernel_h'], params['kernel_w']),
                              padding=(padding_h, padding_w),
                              stride=params['stride_conv'])
        self.conv2 = nn.Conv2d(in_channels=conv1_out_size, out_channels=params['num_filters'],
                               kernel_size=(params['kernel_h'], params['kernel_w']),
                               padding=(padding_h, padding_w),
                               stride=params['stride_conv'])
        self.conv3 = nn.Conv2d(in_channels=conv2_out_size, out_channels=params['num_filters'],
                               kernel_size=(1,1),
                               padding=(0,0),
                               stride=params['stride_conv'])
        self.batchnorm1 = nn.BatchNorm2d(num_features=params['num_channels'])
        self.batchnorm2 = nn.BatchNorm2d(num_features=conv1_out_size)
        self.batchnorm3 = nn.BatchNorm2d(num_features=conv2_out_size)
        self.prelu = nn.PReLU()

    def forward(self, input):
        o1 = self.batchnorm1(input)
        o2 = self.prelu(o1)
        o3 = self.conv1(o2)
        o4 = torch.cat((input, o3), dim=1)
        o5 = self.batchnorm2(o4)
        o6 = self.prelu(o5)
        o7 = self.conv2(o6)
        o8 = torch.cat((input,o3,o7), dim=1)
        o9 = self.batchnorm3(o8)
        o10 = self.prelu(o9)
        out = self.conv3(o10)
        return out


class EncoderBlock(DenseBlock):
    def __init__(self, params):
        super(EncoderBlock, self).__init__(params)
        self.maxpool = nn.MaxPool2d(kernel_size=params['pool'], stride=params['stride_pool'], return_indices=True)

    def forward(self, input):
        out_block = super(EncoderBlock, self).forward(input)
        out_encoder, indices = self.maxpool(out_block)
        return out_encoder, out_block, indices


class DecoderBlock(DenseBlock):
    def __init__(self, params):
        super(DecoderBlock, self).__init__(params)
        self.unpool = nn.MaxUnpool2d(kernel_size=params['pool'], stride=params['stride_pool'])

    def forward(self, input, out_block, indices):
        unpool = self.unpool(input, indices)
        concat = torch.cat((out_block, unpool), dim=1)
        out_block = super(DecoderBlock, self).forward(concat)

        return out_block


class ClassifierBlock(nn.Module):
    def __init__(self, params):
        super(ClassifierBlock, self).__init__()
        self.conv = nn.Conv2d(params['num_channels'], params['num_class'], params['kernel_c'], params['stride_conv'])
        #self.softmax = nn.Softmax2d()

    def forward(self, input):
        out_conv = self.conv(input)
        #out_logit = self.softmax(out_conv)
        return out_conv
