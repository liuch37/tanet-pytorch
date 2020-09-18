'''
This is an example code of applying TANet on any feature maps.
'''

import torch
from backbone.backbone import backbone
from models.TAN import TANet

# input image parameters
batch_size = 2
Height_input = 64
Width_input = 64
Channel_input = 3

input_images = torch.randn(batch_size,Channel_input,Height_input,Width_input)

# feature map extraction
backbone_model = backbone(Channel_input)
feature_maps = backbone_model(input_images)

# TANet
Channel_feature, Height_feature, Width_feature = feature_maps.shape[1], feature_maps.shape[2], feature_maps.shape[3]
tanet_model = TANet(f_in=Channel_feature, C_in=Channel_input, kernel_sizes=[3, 5])
feature_maps_update, att_map = tanet_model(feature_maps, input_images)

print("feature map shape:", feature_maps.shape)
print("updated feature map shape:", feature_maps_update.shape)