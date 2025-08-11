import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange,repeat
import gc
from torchvision.utils import save_image
import sys


def stereo_shift_torch(input_images, depthmaps,sacle_factor=8,shift_both = False,stereo_offset_exponent=1.0):
    '''input: [B, C, H, W] depthmap: [B, H, W]'''

    def _norm_depth(depth,max_val=1):
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (depth - depth_min) / (depth_max - depth_min)
        else:
            out = np.zeros(depth.shape, dtype=depth.dtype)
        return out
    
    def _create_stereo(input_images,depthmaps,sacle_factor,stereo_offset_exponent):
        b, c, h, w = input_images.shape
        derived_image = torch.zeros_like(input_images)
        sacle_factor_px = (sacle_factor / 100.0) * input_images.shape[-1]  #sacle_factor 시차의 크기를 결정하는 비율 (픽셀 이동 크기)
        # 마이너스가 오른쪽시차 플러스가 왼쪽 시차이네 
        
        # filled = torch.zeros(b * h * w, dtype=torch.uint8)
        if True:
            for batch in range(b):
                for row in range(h):
                    # Swipe order should ensure that pixels that are closer overwrite
                    # (at their destination) pixels that are less close
                    for col in range(w) if sacle_factor_px < 0 else range(w - 1, -1, -1):
                        col_d = col + int((depthmaps[batch,row,col] ** stereo_offset_exponent) * sacle_factor_px)
                        if 0 <= col_d < w:
                            derived_image[batch,:,row,col_d] = input_images[batch,:,row,col]
                            # filled[batch * h * w + row * w + col_d] = 1    
                    # stereo_offset_exponent : 깊이 값을 비선형적으로 조정함     
        return derived_image
    depthmaps = _norm_depth(depthmaps)
    
    if shift_both is False: # shift_both 양쪽 시차 모두 생성하게 함 
        left = input_images
        balance = 0
    else:
        balance = 0.5
        left = _create_stereo(input_images,depthmaps,+1 * sacle_factor * balance,stereo_offset_exponent)
    right = _create_stereo(input_images,depthmaps,-1 * sacle_factor * (1 - balance),stereo_offset_exponent)
    return torch.concat([left,right],axis=0)


