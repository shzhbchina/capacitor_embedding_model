import numpy as np
import torch

def random_mask(x, mask_ratio=0.3, mask_value=0.0,mask_chance=0.9):
    # x: [batch, feature_dim]，掩码部分特征
    # mask =1, masked out feature
    if torch.rand(1) <= mask_chance:
        mask = (torch.rand_like(x) < mask_ratio).float()
    else:
        mask = (torch.rand_like(x) < 0.0).float()
    return x * (1-mask) + mask_value * mask, mask
