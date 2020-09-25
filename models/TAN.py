'''
This module is to build TANet integrated model by fusing with any feature map.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import ParameterList
import math
from gradient_layer import cgradient
from gabor_layer import gabor_layer
from attention_layer import attention_layer, attention_layer_light

__all__ = ['Conv_BN_ReLU','TANet']

class Conv_BN_ReLU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0):
        super(Conv_BN_ReLU, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class TANet(nn.Module):
    def __init__(self, f_in, C_in, kernel_sizes=[3], theta_size=4):
        '''
        f_in: feature map number of channels
        C_in: input image number of channels - should be 3 (assume RGB)
        kernel_sizes: list of kernel sizes for Gabor convolution layer
        theta_size: resolution of theta - number of output channesl for Gabor layer
        '''
        super(TANet, self).__init__()
        self.gradient_layer = cgradient(C_in, kernel_size=3)
        self.gabor_layers = nn.Sequential(*[gabor_layer(theta_size, kernel_size=k) for k in kernel_sizes])
        #self.attention_layer = attention_layer(f_in, 1)
        self.attention_layer_light = attention_layer_light(f_in+1+theta_size)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.alphas = ParameterList([Parameter((torch.Tensor([1.0/len(kernel_sizes)]))) for i in range(len(kernel_sizes))])
        self.beta = Parameter(torch.Tensor([1.0]))
        self.kernel_sizes = kernel_sizes
        self.theta_size = theta_size

    def forward(self, feature_map, img):
        img_grad = self.gradient_layer(img) # [batch, 1, H, W]
        img_gabor = [layer(img) for layer in self.gabor_layers] # [batch, theta_size, H, W]
        # weighted sum
        img_gabor[0] = self.alphas[0]*img_gabor[0]
        for i in range(1, len(self.kernel_sizes)):
            img_gabor[0] += self.alphas[i]*img_gabor[i]
        # fusion
        img_fusion = torch.cat((img_gabor[0], self.beta*img_grad), dim=1)

        # customized downsample layer
        img_fusion = self.maxpool(img_fusion)
        img_fusion = self.maxpool(img_fusion)
        
        # concatenate
        img_fusion = torch.cat((feature_map, img_fusion), dim=1)

        # attention
        #feature_map, att_map = self.attention_layer(feature_map, img_fusion)
        feature_map, att_map = self.attention_layer_light(img_fusion)

        return feature_map, att_map

# unit testing
if __name__ == '__main__':
    # fake input feature map and input images
    batch_size = 2
    Height_feature = 160
    Width_feature = 160
    Channel_feature = 512
    Height_input = 640
    Width_input = 640
    Channel_input = 3

    features = torch.randn(batch_size,Channel_feature,Height_feature,Width_feature)
    input_images = torch.randn(batch_size, Channel_input, Height_input, Width_input)

    model = TANet(f_in=Channel_feature, C_in=Channel_input, kernel_sizes=[3, 5], theta_size=4)
    feature_map, att_map = model(features, input_images)

    print(feature_map)
    for name, parameter in model.named_parameters():
        print(name)
    print(model.gabor_layers[0].sigma[0])
    print(model.alphas[0])
    print(model.alphas[1])
    print(model.gradient_layer.weight_x)
    print("Input feature map size is:", features.shape)
    print("Input image size is:", input_images.shape)
    print("Outout feature map size is:", feature_map.shape)
    print("Outout attention map size is:", att_map.shape)