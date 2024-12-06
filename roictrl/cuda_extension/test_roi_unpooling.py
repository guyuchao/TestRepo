from typing import List, Union
import numpy as np
from PIL import Image
import torch
import torch._dynamo
import torch.fx
from torch import nn, Tensor
from torch.jit.annotations import BroadcastingList2
from torch.nn.modules.utils import _pair
from torchvision.extension import _assert_has_ops, _has_ops
from einops import rearrange
import math
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.ops.roi_align import RoIAlign
from torch.utils.cpp_extension import load


roiunpool = load(name="roiunpool", sources=["roiunpool.cpp", "roiunpool.cu"], verbose=True)


def _bilinear_interploate(
    roi_feat,
    y,
    x,
    batch_idx
):
    def is_legal_coord(y, x, roi_height, roi_width):
        if y >= 0 and y < roi_height and x >= 0 and x < roi_width:
            return True
        else:
            return False
  
    _, channels, roi_height, roi_width = roi_feat.shape # torch.Size([160, 320, 7, 7])
    # print(height, width) # 64 64
    
    y_low = int(math.floor(y))
    x_low = int(math.floor(x))
    y_high = int(math.ceil(y))
    x_high = int(math.ceil(x))
   
    ly = y - y_low
    lx = x - x_low
    hy = 1.0 - ly
    hx = 1.0 - lx
    
    w1 = hy * hx if is_legal_coord(y_low, x_low, roi_height, roi_width) else 0
    w2 = hy * lx if is_legal_coord(y_low, x_high, roi_height, roi_width) else 0
    w3 = ly * hx if is_legal_coord(y_high, x_low, roi_height, roi_width) else 0
    w4 = ly * lx if is_legal_coord(y_high, x_high, roi_height, roi_width) else 0
    # print(w1, w2, w3, w4)
    
    feat1 = roi_feat[batch_idx, :, y_low, x_low] if w1 != 0 else 0
    feat2 = roi_feat[batch_idx, :, y_low, x_high] if w2 != 0 else 0
    feat3 = roi_feat[batch_idx, :, y_high, x_low] if w3 != 0 else 0
    feat4 = roi_feat[batch_idx, :, y_high, x_high] if w4 != 0 else 0
    
    val = w1 * feat1 + w2 * feat2 + w3 * feat3 + w4 * feat4
    if 1 - (w1+w2+w3+w4) < 1e-5:
        return val # remove boundary
    else:
        return 0

def _roi_unpooling_cpu(roi_feat, rois, rois_mask, height, width, spatial_scale, aligned):
    
    batch_size, num_rois, channels, roi_height, roi_width = roi_feat.shape # 16 10 320 7 7    
    target_feat = torch.zeros((batch_size, num_rois, channels, height, width), device=roi_feat.device, dtype=roi_feat.dtype)

    roi_feat = rearrange(roi_feat, "b n c h w -> (b n) c h w")
    rois = rearrange(rois, "b n c -> (b n) c")
    rois_mask = rearrange(rois_mask, "b n -> (b n)")
    target_feat = rearrange(target_feat, "b n c h w -> (b n) c h w")
    
    # print(roi_feat.shape, rois_matrix.shape, rois_mask.shape, target_feat.shape) # torch.Size([160, 320, 7, 7]) torch.Size([160, 4]) torch.Size([16, 10]) torch.Size([160, 320, 40, 64])

    roi_feat_ = roi_feat[rois_mask==1]
    rois = rois[rois_mask==1]
    target_feat_ = torch.zeros((roi_feat_.shape[0], channels, height, width), device=roi_feat.device, dtype=roi_feat.dtype)

    offset = 0.5 if aligned else 0.0
    
    for batch_idx in range(roi_feat_.shape[0]):
        x1, y1, x2, y2 = rois[batch_idx]
        
        feat_start_h, feat_start_w = math.ceil(y1 * spatial_scale - offset), math.ceil(x1 * spatial_scale - offset)
        feat_end_h, feat_end_w = math.ceil(y2 * spatial_scale - offset), math.ceil(x2 * spatial_scale - offset)

        feat_height = feat_end_h - feat_start_h
        feat_width = feat_end_w - feat_start_w
        # print(height, width) 160, 256
        
        for h in range(height):
            for w in range(width):
                if h >= feat_start_h and h < feat_end_h and w >= feat_start_w and w < feat_end_w:
                    roi_y = ((h - feat_start_h) / feat_height) * roi_height
                    roi_x = ((w - feat_start_w) / feat_width) * roi_width
                                        
                    target_feat_[batch_idx, :, h, w] = _bilinear_interploate(roi_feat_, roi_y, roi_x, batch_idx)
    
    target_feat[rois_mask==1] = target_feat_
    target_feat = rearrange(target_feat, "(b n) c h w -> b n c h w", b=batch_size)
    return target_feat



if __name__ == "__main__":
    image_list = ["../../datasets/test_image_512/zreal_penguin.png", "../../datasets/test_image_512/ship02.png"]
    feature_map = torch.stack([ToTensor()(Image.open(img).convert("RGB")) for img in image_list])
    feature_map = feature_map.cuda()

    rois = torch.zeros(2, 10, 4)
    rois[0, 0] = torch.tensor([74, 86, 178, 238])
    rois[0, 1] = torch.tensor([174, 89, 283, 277])
    rois[0, 2] = torch.tensor([303, 101, 405, 253])
    rois[0, 3] = torch.tensor([411, 111, 512, 262])
    rois[1, 0] = torch.tensor([192, 21, 299, 209])
    rois = rois.cuda()

    rois_mask = torch.zeros(2, 10)
    rois_mask[0, 0] = 1
    rois_mask[0, 1] = 1
    rois_mask[0, 2] = 1
    rois_mask[0, 3] = 1
    rois_mask[1, 0] = 1
    rois_mask = rois_mask.cuda()

    original_size = [320, 512]
        
    output_size = [61, 61]
    downsample_rate = 2
    spatial_scale = 1 / downsample_rate

    feat_size = [int(l * spatial_scale) for l in original_size]
    feature_map = torch.nn.functional.interpolate(feature_map, size=feat_size, mode='bilinear', align_corners=False)
    
    rois_list = [roi for roi in rois]
    roi_align = RoIAlign(output_size=output_size, spatial_scale=spatial_scale, sampling_ratio=2, aligned=True)

    # first, use original coordinate
    roi_feat = roi_align(feature_map, rois_list) # torch.Size([20, 3, 64, 64])
    roi_feat = rearrange(roi_feat, '(b n) c h w -> b n c h w', b=2)
    # print(roi_feat.shape) # torch.Size([2, 10, 3, 64, 64])
    img = ToPILImage()(roi_feat[0,0])
    img.save("1.jpg")
    '''
    # roi pool directly
    unpool_img = torch.zeros(1, 3, *feat_size)
    
    roi1_x1, roi1_y1, roi1_x2, roi1_y2 = rois[0, 0]
    roi1_x1, roi1_y1, roi1_x2, roi1_y2 = int(roi1_x1*spatial_scale), int(roi1_y1*spatial_scale), int(roi1_x2*spatial_scale), int(roi1_y2*spatial_scale)
    
    unpool_img[:, :, roi1_y1:roi1_y2, roi1_x1:roi1_x2] = torch.nn.functional.interpolate(roi_feat[0, 0:1], size=(roi1_y2-roi1_y1, roi1_x2-roi1_x1), mode='bilinear', align_corners=False)
    unpool_img = ToPILImage()(unpool_img[0])
    unpool_img.save("1_unpool.jpg")
    

    # torch-native
    target_feat = _roi_unpooling(roi_feat, rois, rois_mask, feat_size[0], feat_size[1], spatial_scale, aligned=True)
    unpool_img_my = ToPILImage()(target_feat[0,0])
    unpool_img_my.save("1_myunpool_torch_new.jpg")
    '''

    # cuda
    target_feat = _roi_unpooling_cuda(roi_feat, rois, rois_mask, feat_size[0], feat_size[1], spatial_scale, aligned=True)
    
    unpool_img_my_cuda = ToPILImage()(target_feat[0,0])
    unpool_img_my_cuda.save("1_myunpool_cuda_new.jpg")