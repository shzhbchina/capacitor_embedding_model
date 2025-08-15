import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


def Recon_loss( recon_x, x, pred_z,true_table_ini,disc_table,loss_conf=None):
    pred_x, recon_cont_x, recon_disc_logits_x=recon_x
    data_x, data_cont_x,data_disc_x=x,x[:,~disc_table],x[:,disc_table]
    true_table=true_table_ini.clone()
    true_table=true_table.float()

    # if loss_conf is None:
    #     true_table[:, 0] = 6 * true_table[:, 0]
    #     true_table[:, 1] = 6 * true_table[:, 1]
    #     true_table[:, 12] = 6 * true_table[:, 12]
    #     true_table[:, -50:] = 0.5 * true_table[:, -50:]
    # else:
    #     true_table[:, 0] = loss_conf['volt_weight'] * true_table[:, 0]
    #     true_table[:, 1] = loss_conf['cap_weight'] * true_table[:, 1]
    #     true_table[:, 12] = loss_conf['price_weight'] * true_table[:, 12]
    #     true_table[:, -50:] = loss_conf['ESR_ripple_weight'] * true_table[:, -50:]

    if loss_conf is None:
        reg_weight=50.0
        true_table[:, 0] = 6 * true_table[:, 0]
        true_table[:, 1] = 6 * true_table[:, 1]
        true_table[:, 12] = 6 * true_table[:, 12]
        true_table[:, -50:-25] = 6 * true_table[:, -50:-25]
        weak_norm_wmean=0.0
        weak_norm_wvar=0.0
    else:
        reg_weight=loss_conf['reg_weight']
        true_table[:, 0] = loss_conf['volt_weight'] * true_table[:, 0]
        true_table[:, 1] = loss_conf['cap_weight'] * true_table[:, 1]
        true_table[:, 12] = loss_conf['price_weight'] * true_table[:, 12]
        if loss_conf.get('ESR_ripple_weight') is not None:
            true_table[:, -50:] = loss_conf['ESR_ripple_weight'] * true_table[:, -50:]
        if loss_conf.get('ESR_weight') is not None:
            true_table[:, -50:-25] = loss_conf['ESR_weight'] * true_table[:, -50:-25]
        if loss_conf.get('ripple_weight') is not None:
            true_table[:, -25:] = loss_conf['ripple_weight'] * true_table[:, -25:]
        if loss_conf.get('vol_weight') is not None:
            true_table[:, 3:8] = loss_conf['vol_weight'] * true_table[:, 3:8]
        if loss_conf.get('weak_norm_wmean') is not None:
            weak_norm_wmean = loss_conf['weak_norm_wmean']
        if loss_conf.get('weak_norm_wvar') is not None:
            weak_norm_wvar = loss_conf['weak_norm_wvar']

    cont_mask = true_table[:, ~disc_table]     # [batch, num_cont_features]
    disc_mask = true_table[:, disc_table]      # [batch, num_disc_features]
    loss_cont = ((recon_cont_x - data_cont_x) ** 2 * cont_mask).sum() / (cont_mask.sum() + 1e-8)
    loss_cat = 0
    num_disc = disc_mask.shape[1] if len(disc_mask.shape) > 1 else 1
    for i, logits in enumerate(recon_disc_logits_x):
        mask_col = disc_mask[:, i]
        ce = F.cross_entropy(logits, data_disc_x[:, i].long(), reduction='none')
        loss_cat += (ce * mask_col).sum() / (mask_col.sum() + 1e-8)
    loss_cat = loss_cat / num_disc
    recon_loss = loss_cont + loss_cat

    # --------- 弱正则：z 的中心 / 方差 ----------
    z_mean = pred_z.mean(dim=0)                     # [Dz]
    z_var  = pred_z.var(dim=0, unbiased=False)      # [Dz]
    z_center_loss = (z_mean ** 2).mean()
    z_var_loss    = ((z_var - 1.0) ** 2).mean()
    weak_norm_loss=weak_norm_wmean*z_center_loss+weak_norm_wvar*z_var_loss


    return [
        (recon_loss.mean(dim=0) +weak_norm_loss),
        (recon_loss).mean(dim=0),
        weak_norm_loss,
    ]
