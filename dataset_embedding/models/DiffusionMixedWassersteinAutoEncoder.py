import torch
import torch.nn as nn
import numpy as np

class GaussianDiffusionScheduler:
    def __init__(self, timesteps=100, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.timesteps = timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        t_ = t.view(-1, 1).to(x_start.device)
        sqrt_alpha_bar = self.alpha_bars[t_].sqrt()
        sqrt_one_minus = (1 - self.alpha_bars[t_]).sqrt()
        return sqrt_alpha_bar * x_start + sqrt_one_minus * noise

# ----- Diffusion MLP Decoder -----
class DiffusionMLPDecoder(nn.Module):
    def __init__(self,in_dim, latent_dim, hidden_dim, t_embed_dim=16):
        super().__init__()
        self.t_embed = nn.Embedding(100, t_embed_dim)  # timesteps=100
        self.input_proj = nn.Linear(in_dim + t_embed_dim + latent_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim)
        )

    def forward(self, x_noisy,t,z):
        t_emb = self.t_embed(t)
        inp = torch.cat([x_noisy,t_emb,z], dim=1)
        h = self.input_proj(inp)
        out = self.net(h)
        return out  # (B, out_dim): 预测噪声

class DiffusionMixedWAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, disc_dim, discrete_feature_list, diffusion_timesteps=100):
        super().__init__()
        self.discrete_feature_list = discrete_feature_list
        disc_true_table = discrete_feature_list['true_table']
        tot_discs = np.sum(disc_true_table.astype(int))
        tot_embedding_dims = np.sum(discrete_feature_list['dimensions']['embedding_dim']).astype(int)
        # WAE Encoder：输出deterministic z
        self.encoder = nn.Sequential(
            nn.Linear(input_dim - tot_discs + tot_embedding_dims, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        self.embedding_layers = nn.ModuleList()
        embedding_dimensions = discrete_feature_list['dimensions']
        for num_categories, embedding_size in zip(embedding_dimensions['cat_num'], embedding_dimensions['embedding_dim']):
            self.embedding_layers.append(nn.Embedding(num_categories, embedding_size))

        num_cont_features = input_dim - tot_discs
        self.diffusion_decoder = DiffusionMLPDecoder(
            in_dim=num_cont_features,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            t_embed_dim=16
        )
        self.diffusion_timesteps = diffusion_timesteps
        self.scheduler = GaussianDiffusionScheduler(timesteps=diffusion_timesteps)

        self.decoder_disc_layers = nn.ModuleList()
        for category_size in embedding_dimensions['cat_num']:
            layer = nn.Sequential(
                nn.Linear(latent_dim, disc_dim),
                nn.LayerNorm(disc_dim),
                nn.ReLU(),
                nn.Linear(disc_dim, category_size),
            )
            self.decoder_disc_layers.append(layer)

    def encode(self, x_cat):
        z = self.encoder(x_cat)
        return z

    def forward(self, x, t=None, training_diffusion=True):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        disc_true_table = self.discrete_feature_list['true_table']
        disc_idx = np.where(disc_true_table)[0]
        embedding_vectors = []
        for i, layer in enumerate(self.embedding_layers):
            input_column = x[:, disc_idx[i]]
            embedded_feature = layer(input_column.long())
            embedding_vectors.append(embedded_feature)
        embedding_disc = torch.cat(embedding_vectors, dim=1)
        x_cat = torch.cat([x[:, ~disc_true_table], embedding_disc], dim=1)

        # 1. 编码z
        z = self.encode(x_cat)

        # 2. Diffusion阶段
        if training_diffusion:
            B = x.shape[0]
            if t is None:
                t = torch.randint(0, self.diffusion_timesteps, (B,), device=x.device)
            noise = torch.randn_like(x[:, ~disc_true_table])
            x_noisy = self.scheduler.q_sample(x[:, ~disc_true_table], t, noise)
            #pred_noise = self.diffusion_decoder(z, t) #pred noise at t
            pred_noise = self.diffusion_decoder(x_noisy,t,z)

            t_ = t.view(-1, 1)
            sqrt_alpha_bar = self.scheduler.alpha_bars[t_].sqrt().to(x.device)
            sqrt_one_minus = (1 - self.scheduler.alpha_bars[t_]).sqrt().to(x.device)
            x0_pred = (x_noisy - sqrt_one_minus * pred_noise) / sqrt_alpha_bar

            # 离散输出
            logits_vectors, decode_vectors = [], []
            for i, layer in enumerate(self.decoder_disc_layers):
                disc_feature = layer(z)
                single_cat = torch.argmax(disc_feature, dim=1)
                logits_vectors.append(disc_feature)
                decode_vectors.append(single_cat)
            recon_disc_x = torch.stack(decode_vectors, dim=1)
            batch_size = z.shape[0]
            total_features = len(disc_true_table)
            pred_x = torch.zeros(batch_size, total_features, device=z.device)
            pred_x[:, disc_true_table] = recon_disc_x.float()
            pred_x[:, ~disc_true_table] = x0_pred

            return [[pred_x,x0_pred , logits_vectors], z,  noise,pred_noise,x_noisy]
        else:
            raise NotImplementedError("Sampling in diffusion not implemented in forward. Use sample() separately.")

    @torch.no_grad()
    def decode(self, z, shape=None):
        """
        z: (B, latent_dim)
        shape: (B, num_cont_features)  # 即input_dim - tot_discs
        返回：和decode一样的 [pred_x, recon_cont_x, recon_disc_x]
        """
        batch_size = z.shape[0]
        device = z.device
        # 推断连续变量
        if shape is None:
            # 自动推断连续维度
            disc_true_table = self.discrete_feature_list['true_table']
            num_cont_features = (~disc_true_table).sum()
            shape = (batch_size, num_cont_features)
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.diffusion_timesteps)):
            t_fill = torch.full((batch_size,), t, device=device, dtype=torch.long)
            pred_noise = self.diffusion_decoder(z, t_fill)
            pred_noise = self.diffusion_decoder(x, t_fill, z)
            alpha = self.scheduler.alphas[t]
            alpha_bar = self.scheduler.alpha_bars[t]
            beta = self.scheduler.betas[t]
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
            x = 1 / alpha.sqrt() * (x - (1 - alpha) / (1 - alpha_bar).sqrt() * pred_noise) + beta.sqrt() * noise
        recon_cont_x = x  # (B, num_cont_features)

        # 推断离散变量
        logits_vectors = []
        decode_vectors = []
        for i, layer in enumerate(self.decoder_disc_layers):
            disc_feature = layer(z)
            single_cat = torch.argmax(disc_feature, dim=1)
            logits_vectors.append(disc_feature)
            decode_vectors.append(single_cat)
        recon_disc_x = torch.stack(decode_vectors, dim=1)  # (B, num_discrete_vars)

        # 组装完整输出，与decode接口完全一致
        disc_true_table = self.discrete_feature_list['true_table']
        total_features = len(disc_true_table)
        pred_x = torch.zeros(batch_size, total_features, device=z.device)
        pred_x[:, disc_true_table] = recon_disc_x.float()
        pred_x[:, ~disc_true_table] = recon_cont_x

        return [pred_x, recon_cont_x, recon_disc_x]

# def compute_mmd(z, z_prior, sigma=1.0):
#     """z, z_prior: (B, latent_dim)"""
#     # 高斯核
#     zz = z.unsqueeze(1) - z.unsqueeze(0)
#     zz_prior = z_prior.unsqueeze(1) - z_prior.unsqueeze(0)
#     Kzz = torch.exp(- (zz ** 2).sum(2) / (2 * sigma ** 2))
#     Kpp = torch.exp(- (zz_prior ** 2).sum(2) / (2 * sigma ** 2))
#     Kzp = torch.exp(- ((z.unsqueeze(1) - z_prior.unsqueeze(0)) ** 2).sum(2) / (2 * sigma ** 2))
#     mmd = Kzz.mean() + Kpp.mean() - 2 * Kzp.mean()
#     return mmd

# # === 训练 ===
# outputs, z, noise, pred_noise, t, mmd_loss, x0_pred, x_cont_gt = model(x, training_diffusion=True)
# loss_diff = F.mse_loss(pred_noise, noise)
# loss_x0 = F.mse_loss(x0_pred, x_cont_gt)
# total_loss = loss_diff + 0.5 * loss_x0 + 10.0 * mmd_loss   # MMD权重建议5-50，可微调
# total_loss.backward()
# optimizer.step()
#
# # === 采样 ===
# z = torch.randn(B, latent_dim).to(device)   # 采样prior生成
# samples = model.sample(z, shape=(B, input_dim - tot_discs))
