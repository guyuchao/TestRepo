from PIL import Image
import torch
import torch._dynamo
import torch.fx
from einops import rearrange
import math
from torch.utils.cpp_extension import load
import torchvision
from torchvision.transforms.transforms import ToTensor

roiunpool = load(name="roiunpool", sources=["dyna/cuda_extension/roiunpool.cpp", "dyna/cuda_extension/roiunpool.cu"], verbose=True)
# roiunpool = load(name="roiunpool", sources=["roiunpool.cpp", "roiunpool.cu"], verbose=True)


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
    

def _roi_unpooling(roi_feat, rois, rois_mask, height, width, spatial_scale, aligned):
    
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


def _roi_unpooling_cuda(roi_feat, rois, rois_mask, height, width, spatial_scale, aligned):
    batch_size, num_rois, channels, roi_height, roi_width = roi_feat.shape
    target_feat = torch.zeros((batch_size, num_rois, channels, height, width), device=roi_feat.device, dtype=roi_feat.dtype)
    
    roi_feat = rearrange(roi_feat, 'b n c h w -> (b n) c h w') # 20, 3, 64, 64
    rois = rearrange(rois, "b n c -> (b n) c") # 20, 4
    rois_mask = rearrange(rois_mask, "b n -> (b n)") # 2, 10
    target_feat = rearrange(target_feat, "b n c h w -> (b n) c h w")
    
    roi_feat_ = roi_feat[rois_mask==1]
    rois_ = rois[rois_mask==1] 
    
    target_feat_ = Roi_Unpool.apply(roi_feat_, rois_, spatial_scale, height, width, roi_height, roi_width, aligned)
    
    target_feat[rois_mask==1] = target_feat_
    target_feat = rearrange(target_feat, "(b n) c h w -> b n c h w", b=batch_size)
    
    return target_feat

class Roi_Unpool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, roi_feat, rois, spatial_scale, height, width, pooled_height, pooled_width, aligned):

        target_feat = roiunpool.roi_unpool_forward(roi_feat, rois, spatial_scale, height, width, pooled_height, pooled_width, aligned)

        ctx.save_for_backward(rois)
        ctx.height, ctx.width, ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale, ctx.aligned = height, width, pooled_height, pooled_width, spatial_scale, aligned

        return target_feat

    @staticmethod
    def backward(ctx, grad):
        rois = ctx.saved_tensors[0]
        grad_input = roiunpool.roi_unpool_backward(grad.contiguous(), rois, ctx.spatial_scale, ctx.height, ctx.width, ctx.pooled_height, ctx.pooled_width, ctx.aligned)
        return grad_input, None, None, None, None, None, None, None


def prepare_input_data():
    image_list = ["../../datasets/test_image_512/zreal_penguin.png", "../../datasets/test_image_512/ship02.png"]
    feature_map = torch.stack([ToTensor()(Image.open(img).convert("RGB")) for img in image_list])
    feature_map = feature_map.cuda().requires_grad_()

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
    return feature_map, rois, rois_mask

if __name__ == "__main__":
    feature_map, rois, rois_mask = prepare_input_data()
    # print(feature_map.shape, feature_map.requires_grad, rois.shape, rois.requires_grad, rois_mask.shape, rois_mask.requires_grad)
    # torch.Size([2, 3, 320, 512]) True torch.Size([2, 10, 4]) False torch.Size([2, 10]) False

    original_size = [320, 512]
    output_size = [61, 61]
    downsample_rate = 2
    spatial_scale = 1 / downsample_rate

    feat_size = [int(l * spatial_scale) for l in original_size]
    feature_map_ = torch.nn.functional.interpolate(feature_map, size=feat_size, mode='bilinear', align_corners=False)

    rois_list = [roi for roi in rois]

    # first, use original coordinate
    rois_feat = torchvision.ops.roi_align(feature_map_, rois_list, spatial_scale=spatial_scale, output_size=output_size, sampling_ratio=2, aligned=True) # torch.Size([20, 3, 64, 64])
    rois_feat = rearrange(rois_feat, '(b n) c h w -> b n c h w', b=2)
    # print(rois_feat.shape) # torch.Size([2, 10, 3, 64, 64])
    
    # torch-native
    unpool_type = "cuda"
    if unpool_type == "torch":
        target_feat = _roi_unpooling(rois_feat, rois, rois_mask, feat_size[0], feat_size[1], spatial_scale, aligned=True)
        loss = target_feat.sum()
        print(loss) # tensor(35164.1797, device='cuda:0', grad_fn=<SumBackward0>)
        loss.backward()
        print(feature_map.grad.sum()) # tensor(64503., device='cuda:0')
    else:
        target_feat = _roi_unpooling_cuda(rois_feat, rois, rois_mask, feat_size[0], feat_size[1], spatial_scale, aligned=True)
        loss = target_feat.sum()
        print(loss) # tensor(35164.1797, device='cuda:0', grad_fn=<SumBackward0>)
        loss.backward()
        print(feature_map.grad.sum()) # tensor(64503., device='cuda:0')