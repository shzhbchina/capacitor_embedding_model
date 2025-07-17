import torch
import torch.nn as nn
def VAE_loss(model_output,train_data,true_table):
    [recon_x, mu, logvar] = model_output

    BCE = nn.functional.mse_loss(recon_x*true_table, train_data*true_table, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return [BCE + KLD,BCE,KLD]


