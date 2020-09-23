'''
This model is to build a attention layer.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['attention_layer', 'attention_layer_light']

class attention_layer(nn.Module):
    def __init__(self, in_dim, texture_dim):
        super(attention_layer, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=texture_dim, out_channels=in_dim//8 , kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8 , kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.final_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax  = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, z):
        '''
        input:
        x: feature maps (batch, C_in, H, W)
        z: texture images (batch, C_texture, H, W)
        output:
        out: gamma*attention value + input feature maps (batch, C_in, H, W)
        attention: (batch, N, N), N=W*H
        '''
        batch, C, W ,H = x.size()
        proj_query  = self.query_conv(z).view(batch,-1,W*H).permute(0,2,1) # (batch, C, N) -> (batch, N, C)
        proj_key =  self.key_conv(x).view(batch,-1,W*H) # (batch, C, N)
        energy =  torch.bmm(proj_query,proj_key) # batch matrix by matrix multiplication
        attention = self.softmax(energy) # (batch, N, N) 
        proj_value = self.value_conv(x).view(batch,-1,W*H) # (batch, C, N)

        out = torch.bmm(proj_value,attention.permute(0,2,1)) # (batch, C, N)
        out = out.view(batch,C,W,H)
        out = self.final_conv(out) # (batch C, W, H)
        out = self.gamma*out + x

        return out, attention

class attention_layer_light(nn.Module):
    def __init__(self, in_dim, texture_dim):
        super(attention_layer_light, self).__init__()
        self.conv1 = nn.Conv2d(texture_dim, in_dim, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_dim, 1, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.in_dim = in_dim

    def forward(self, x, z):
        '''
        input:
        x: feature maps (batch, C_in, H, W)
        z: texture images (batch, C_texture, H, W)
        output:
        out: gamma*attention value + input feature maps (batch, C_in, H, W)
        attention: (batch, H, W)
        '''
        x_origin = x
        z = self.conv1(z) # (batch, C_in, H, W)
        x = self.conv2(x) # (batch, C_in, H, W)
        combine = self.conv3(torch.tanh(x + z)) # [batch, 1, H, W]
        combine_flat = combine.view(combine.size(0), -1) # resize to [batch, H*W]
        attention_weights = self.softmax(combine_flat) # [batch, H*W]
        attention_weights = attention_weights.view(combine.size()) # [batch, 1, H, W]
        glimpse = x_origin * attention_weights.repeat(1, self.in_dim, 1, 1) # [batch, C_in, H, W]
        out = self.gamma * glimpse + x_origin # [batch, C_in, H, W]

        return out, attention_weights

# unit testing
if __name__ == '__main__':
    # fake input feature map
    batch_size = 2
    Height = 32
    Width = 32
    Channel = 128
    texture_dim = 1

    features = torch.randn(batch_size,Channel,Height,Width)
    texture_image = torch.randn(batch_size, texture_dim, Height, Width)
    attn_layer = attention_layer(Channel, texture_dim)
    new_features, att_map = attn_layer(features, texture_image)
    print(new_features)
    print("shape of new feature map is:", new_features.shape)
    print("shape of attention map is:", att_map.shape)

    batch_size = 16
    Height = 160
    Width = 160
    Channel = 512
    texture_dim = 1

    features = torch.randn(batch_size,Channel,Height,Width)
    texture_image = torch.randn(batch_size, texture_dim, Height, Width)
    attn_layer = attention_layer_light(Channel, texture_dim)
    new_features, att_map = attn_layer(features, texture_image)
    print(new_features)
    print(att_map[0][0].sum())
    print("shape of new feature map is:", new_features.shape)
    print("shape of attention map is:", att_map.shape)