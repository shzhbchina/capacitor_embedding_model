import torch

def mae_loss(recon, x, mask, embedding, alpha=1.0, beta=0.1):
    # 1. 只对mask部分计算重建误差
    mse_loss = ((recon - x)[~mask]).pow(2).mean()
    # 2. embedding均匀分布约束（如最大熵/球面分布/拉普拉斯散度，举例用方差+均值为0约束）
    embed_mean = embedding.mean(dim=0)
    embed_var = embedding.var(dim=0)
    uniform_loss = (embed_mean.pow(2).mean() + (embed_var - 1).pow(2).mean())
    # 3. 总损失
    loss = alpha * mse_loss + beta * uniform_loss
    return loss
