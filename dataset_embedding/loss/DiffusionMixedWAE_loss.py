import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


def DiffusionMixedWAE_loss( recon_x, x, z,noise_x_pred,noise_x,true_table,disc_table):
    pred_x, recon_cont_x, recon_disc_logits_x=recon_x
    data_x, data_cont_x,data_disc_x=x,x[:,~disc_table],x[:,disc_table]
    z_prior = torch.randn_like(z)
    reconstruction_loss_scale=1.0
    kernel_choice='imq'
    reg_weight=10.0
    recon_weight=0.5
    N = z.shape[0]  # batch size
    true_table[:,0]=6*true_table[:,0]
    true_table[:, 1] = 6 * true_table[:, 1]
    cont_dim=data_cont_x.shape[1]


    loss_cont = F.mse_loss(recon_cont_x, data_cont_x)
    loss_cont_noise=F.mse_loss(noise_x_pred, noise_x)
    loss_cat = 0
    for i, logits in enumerate(recon_disc_logits_x):
        loss_cat += F.cross_entropy(logits, data_disc_x[:, i].long())
    recon_loss = recon_weight*loss_cont + loss_cont_noise+loss_cat


    if kernel_choice == "rbf":
        k_z = rbf_kernel(z, z)
        k_z_prior = rbf_kernel(z_prior, z_prior)
        k_cross = rbf_kernel(z, z_prior)
    else:
        k_z = imq_kernel(z, z)
        k_z_prior = imq_kernel(z_prior, z_prior)
        k_cross = imq_kernel(z, z_prior)

    mmd_z = (k_z - k_z.diag().diag()).sum() / ((N - 1) * N)
    mmd_z_prior = (k_z_prior - k_z_prior.diag().diag()).sum() / ((N - 1) * N)
    mmd_cross = k_cross.sum() / (N ** 2)

    mmd_loss = mmd_z + mmd_z_prior - 2 * mmd_cross

    return [
        (recon_loss.mean(dim=0) + reg_weight * mmd_loss)/cont_dim,
        (recon_loss).mean(dim=0)/cont_dim,
        mmd_loss/cont_dim,
    ]


def imq_kernel( z1, z2):
    """Returns a matrix of shape [batch x batch] containing the pairwise kernel computation"""
    latent_dim=z1.shape[1]
    kernel_bandwidth=1.0
    kernel_bandwidth = calculate_median_bandwidth(z1.detach())
    scales=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    Cbase = (
            2.0 * latent_dim * kernel_bandwidth ** 2
    )
    k = 0

    for scale in scales:
        C = scale * Cbase
        k += C / (C + torch.norm(z1.unsqueeze(1) - z2.unsqueeze(0), dim=-1) ** 2)

    return k


def rbf_kernel(z1, z2):
    """Returns a matrix of shape [batch x batch] containing the pairwise kernel computation"""
    latent_dim = z1.shape[1]
    kernel_bandwidth=1.0
    C = 2.0 * latent_dim * kernel_bandwidth ** 2

    k = torch.exp(-torch.norm(z1.unsqueeze(1) - z2.unsqueeze(0), dim=-1) ** 2 / C)

    return k


def calculate_median_bandwidth(z: torch.Tensor):
    """
    使用中位數啟發法計算核函數帶寬。

    Args:
        z (torch.Tensor): 輸入的潛在變數批次，形狀 [N, D]。

    Returns:
        float: 計算出的帶寬值。
    """
    # 1. 高效計算兩兩之間的歐氏距離
    # pdist 會返回一個包含所有 i < j 的點對 dist(z_i, z_j) 的一維向量
    pairwise_distances = torch.pdist(z, p=2)

    # 2. 找到距離的中位數
    # 如果距離為0，可能會導致問題，所以加上一個極小值
    median_dist = torch.median(pairwise_distances)

    # 帶寬通常是中位數或其倍數，這裡我們直接用中位數
    # .item() 將 tensor 轉換為 python 純數字
    bandwidth = median_dist.item()

    # 避免帶寬為0的情況
    if bandwidth == 0:
        bandwidth = 1.0  # 如果所有點都重合，給一個預設值

    return bandwidth