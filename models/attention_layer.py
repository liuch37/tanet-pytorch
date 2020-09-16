'''
This model is to build a attention layer.
'''

import torch
import torch.nn as nn

__all__ = ['attention_layer']

class attention_layer(nn.Module):
    def __init__(self, in_dim):
        super(attention_layer, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8 , kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8 , kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.final_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax  = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        '''
        input:
        x: feature maps (batch, C, W, H)
        output:
        out: gamma*attention value + input feature maps
        attention: (batch, N, N), N=W*H
        '''
        batch, C, W ,H = x.size()
        proj_query  = self.query_conv(x).view(batch,-1,W*H).permute(0,2,1) # (batch, C, N) -> (batch, N, C)
        proj_key =  self.key_conv(x).view(batch,-1,W*H) # (batch, C, N)
        energy =  torch.bmm(proj_query,proj_key) # batch matrix by matrix multiplication
        attention = self.softmax(energy) # (batch, N, N) 
        proj_value = self.value_conv(x).view(batch,-1,W*H) # (batch, C, N)

        out = torch.bmm(proj_value,attention.permute(0,2,1)) # (batch, C, N)
        out = out.view(batch,C,W,H)
        out = self.final_conv(out) # (batch C, W, H)
        out = self.gamma*out + x

        return out, attention

# unit testing
if __name__ == '__main__':
    # fake input feature map
    batch_size = 2
    Height = 24
    Width = 32
    Channel = 128

    features = torch.randn(batch_size,Channel,Height,Width)
    attn_layer = attention_layer(Channel)
    new_features, att_map = attn_layer(features)
    print(new_features)
    print("shape of new feature map is:", new_features.shape)
    print("shape of attention map is:", att_map.shape)