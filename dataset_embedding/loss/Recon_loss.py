import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


def Recon_loss( recon_x, x, true_table,disc_table,loss_conf=None):
    pred_x, recon_cont_x, recon_disc_logits_x=recon_x
    data_x, data_cont_x,data_disc_x=x,x[:,~disc_table],x[:,disc_table]
    true_table=true_table.float()

    if loss_conf is None:
        true_table[:, 0] = 6 * true_table[:, 0]
        true_table[:, 1] = 6 * true_table[:, 1]
        true_table[:, 12] = 6 * true_table[:, 12]
        true_table[:, -50:] = 0.5 * true_table[:, -50:]
    else:
        true_table[:, 0] = loss_conf['volt_weight'] * true_table[:, 0]
        true_table[:, 1] = loss_conf['cap_weight'] * true_table[:, 1]
        true_table[:, 12] = loss_conf['price_weight'] * true_table[:, 12]
        true_table[:, -50:] = loss_conf['ESR_ripple_weight'] * true_table[:, -50:]

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

    return [
        (recon_loss.mean(dim=0) +0.0),
        (recon_loss).mean(dim=0),
        torch.tensor(0.0),
    ]
