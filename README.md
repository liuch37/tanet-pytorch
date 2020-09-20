# TANet
The official implementation of Texture-Aware Attention Network (TANet) with PyTorch >= v1.4.0.

## Task

- [x] Color-Gradient layer
- [x] Gabor layer
- [x] Attention layer
- [x] TANet model
- [x] Example

## Usage
1) Get any feature maps.

2) Feed feature maps and input images to TANet.

Here is an example:
```
from models.TAN import TANet

Channel_feature, Height_feature, Width_feature = feature_maps.shape[1], feature_maps.shape[2], feature_maps.shape[3]
tanet_model = TANet(f_in=Channel_feature, C_in=Channel_input, kernel_sizes=[3, 5])
feature_maps_update, att_map = tanet_model(feature_maps, input_images)
```

## Test
Run the below script
```
python example.py
```

## Source
1. Test image from IIIT5K dataset: https://github.com/ocr-algorithm-and-data/IIIT5K/tree/master/test
