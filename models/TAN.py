'''
This module is to build TANet integrated model by fusing with any feature map.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import ParameterList
from .gradient_layer import cgradient
from .gabor_layer import gabor_layer
from .attention_layer import attention_layer

__all__ = ['TANet']

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

        # resize
        feature_map = F.interpolate(img_fusion, size=(feature_map.shape[2],feature_map.shape[3]), mode='bilinear')

        '''
        # attention
        feature_map, att_map = self.attention_layer(feature_map, img_fusion)
        '''

        return feature_map

# unit testing
if __name__ == '__main__':
    # fake input feature map and input images
    batch_size = 2
    Height_feature = 160
    Width_feature = 160
    Channel_feature = 128
    Height_input = 640
    Width_input = 640
    Channel_input = 3

    features = torch.randn(batch_size,Channel_feature,Height_feature,Width_feature)
    input_images = torch.randn(batch_size, Channel_input, Height_input, Width_input)

    model = TANet(f_in=Channel_feature, C_in=Channel_input, kernel_sizes=[3, 5], theta_size=4)
    feature_map = model(features, input_images)

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