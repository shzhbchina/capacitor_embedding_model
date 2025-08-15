import torch
import torch.nn as nn
import numpy as np

from dataset_embedding.models.PT_MixedWassersteinAutoEncoder import MaskedMixedWAE
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
class ResidualRefiner(nn.Module):
    def __init__(self, zcond_dim, out_dim, hidden=512, p=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(zcond_dim + out_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(p),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(p),
            nn.Linear(hidden, out_dim)
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, z_cond, x_hat):
        delta = self.net(torch.cat([z_cond, x_hat], dim=1))
        return x_hat + self.alpha * delta

class RCMainMaskedMixedWAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, disc_dim, discrete_feature_list, dropout=0.001,encoder_path=None,non_refine_path=None,detach_refine=False):
        super(RCMainMaskedMixedWAE, self).__init__()
        self.discrete_feature_list = discrete_feature_list

        disc_true_table = discrete_feature_list['true_table']
        tot_discs = np.sum(disc_true_table.astype(int))
        tot_embedding_dims = np.sum(discrete_feature_list['dimensions']['embedding_dim']).astype(int)
        self.input_dim = input_dim  # 原始特征数

        eh_hidden_dim=32 * 8
        eh_latent_dim=16 * 4
        eh_disc_dim=4 * 8
        self.encoder_head=MaskedMixedWAE(input_dim, eh_hidden_dim, eh_latent_dim, eh_disc_dim, discrete_feature_list, dropout=0.0)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if encoder_path is not None:
            self.encoder_head.load_state_dict(torch.load(encoder_path,map_location=device))
        for param in self.encoder_head.parameters():
            param.requires_grad = False


        # 其余结构和你第二阶段完全一样
        self.encoder = nn.Sequential(
            nn.Linear(eh_latent_dim, hidden_dim),  # 多一倍输入
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )

        self.discrete_feature_list = discrete_feature_list
        self.decoder_disc_layers = nn.ModuleList()
        embedding_dimensions = discrete_feature_list['dimensions']
        for category_size in embedding_dimensions['cat_num']:
            layer = nn.Sequential(
                nn.Linear(latent_dim, disc_dim),
                nn.LayerNorm(disc_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(disc_dim, disc_dim),
                nn.LayerNorm(disc_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(disc_dim, category_size)
            )
            self.decoder_disc_layers.append(layer)




        #   也可以直接拼 one-hot，但 embedding 更紧凑
        self.use_embed = True     # ← 想直接用 one-hot 则改为 False
        self.disc_emb_layers = nn.ModuleList()
        total_disc_embed_dim = 0
        for cat_size, embed_dim in zip(embedding_dimensions['cat_num'],
                                       embedding_dimensions['embedding_dim']):
            if self.use_embed:
                self.disc_emb_layers.append(nn.Embedding(cat_size, embed_dim))
                total_disc_embed_dim += embed_dim
            else:
                self.disc_emb_layers.append(None)
                total_disc_embed_dim += cat_size   # one-hot 长度

        cont_in_dim = latent_dim + total_disc_embed_dim
        self.decoder_cont = nn.Sequential(
            nn.Linear(cont_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim - tot_discs)
        )

        linear_layers = [m for m in self.decoder_cont.modules() if isinstance(m, nn.Linear)]
        if len(linear_layers) >= 2:
            # 倒数第2层、倒数第1层
            linear_layers[-2] = spectral_norm(linear_layers[-2])
            #linear_layers[-1] = spectral_norm(linear_layers[-1])
        #torch.nn.utils.remove_spectral_norm(module)

        self.refine_steps = 2  # 可以先试 2 或 3
        self.refiner = ResidualRefiner(
            zcond_dim=cont_in_dim,
            out_dim=(input_dim - tot_discs),
            hidden=hidden_dim,
            p=dropout  # dropout ≤0.1
        )


        if non_refine_path is not None:
            self.load_state_dict(torch.load(non_refine_path,map_location=device),strict=False)
        self.detach_refine=detach_refine


    def forward(self, x, tau=1.0,deterministic=False):
        """tau 为 gumbel_softmax 温度"""
        self.train(not deterministic)  # 自动切换模式
        z = self.encode(x)

        disc_true_table = self.discrete_feature_list['true_table']

        logits_list, oh_embed_list = [], []
        for i, layer in enumerate(self.decoder_disc_layers):
            logits = layer(z)                    # (B, cat_i)
            logits_list.append(logits)
            # prob = F.gumbel_softmax(logits, tau=tau, hard=False)


            prob = F.softmax(logits, dim=-1)

            if self.use_embed:
                # index = prob.argmax(dim=1)      # (B,)
                # emb = self.disc_emb_layers[i](index)  # (B, embed_dim)
                emb_matrix = self.disc_emb_layers[i].weight  # (cat, embed_dim)
                emb = prob @ emb_matrix
            else:
                emb = prob                      # 直接用soft one-hot
            oh_embed_list.append(emb)

        disc_emb_cat = torch.cat(oh_embed_list, dim=1)  # (B, total_disc_embed_dim)
        disc_emb_cat_detached = disc_emb_cat.detach().clone()
        z_cond = torch.cat([z, disc_emb_cat_detached], dim=1)    # (B, latent+disc_emb)
        #z_cond = torch.cat([z, disc_emb_cat], dim=1)    # (B, latent+disc_emb)
        #z_cond = torch.cat([z, disc_emb_cat], dim=1)
        recon_cont_x = self.decoder_cont(z_cond)
        # 多步 refinement 循环
        recon_steps = [recon_cont_x]
        for t in range(self.refine_steps - 1):
            x_in = recon_steps[-1].detach() if self.detach_refine else recon_steps[-1]
            recon_steps.append(self.refiner(z_cond, x_in))
        # 最终输出最后一步，也可以把整个 recon_steps 传到 loss 里
        recon_cont_x_final = recon_steps[-1]

        recon_disc_x = torch.stack(
            [logit.argmax(dim=1) for logit in logits_list],
            dim=1
        )

        B, total_feat = z.size(0), len(disc_true_table)
        pred_x = torch.zeros(B, total_feat, device=z.device)
        pred_x[:, disc_true_table] = recon_disc_x.float()
        #pred_x[:, ~disc_true_table] = recon_cont_x
        pred_x[:, ~disc_true_table] = recon_cont_x_final
        return [[pred_x, recon_cont_x_final, logits_list, recon_steps], z]

        #return [[pred_x, recon_cont_x, logits_list], z]

    def decode(self, z,tau=1.0):
        # 跟第二阶段完全一致
        self.eval()
        disc_true_table = self.discrete_feature_list['true_table']
        with torch.no_grad():
            logits_list, oh_embed_list = [], []
            for i, layer in enumerate(self.decoder_disc_layers):
                logits = layer(z)                    # (B, cat_i)
                logits_list.append(logits)

                # ① 生成可微 one-hot  (hard=False 更稳定，也可 hard=True + straight-through)
                #prob = F.gumbel_softmax(logits, tau=tau, hard=False)
                prob = F.softmax(logits, dim=-1)

                if self.use_embed:
                    # index = prob.argmax(dim=1)      # (B,)
                    # emb = self.disc_emb_layers[i](index)  # (B, embed_dim)
                    emb_matrix = self.disc_emb_layers[i].weight  # (cat, embed_dim)
                    emb = prob @ emb_matrix
                else:
                    emb = prob                      # 直接用soft one-hot

                oh_embed_list.append(emb)

            disc_emb_cat = torch.cat(oh_embed_list, dim=1)  # (B, total_disc_embed_dim)
            z_cond = torch.cat([z, disc_emb_cat], dim=1)    # (B, latent+disc_emb)
            recon_cont_x = self.decoder_cont(z_cond)
            recon_steps = [recon_cont_x]
            for t in range(self.refine_steps - 1):
                # x_in = recon_steps[-1]
                # recon_steps.append(self.refiner(z_cond, x_in))
                x_in = recon_steps[-1].detach() if self.training else recon_steps[-1]
                recon_steps.append(self.refiner(z_cond, x_in))
            # 最终输出最后一步，也可以把整个 recon_steps 传到 loss 里
            recon_cont_x_final = recon_steps[-1]

            recon_disc_x = torch.stack(
                [logit.argmax(dim=1) for logit in logits_list],
                dim=1
            )

            B, total_feat = z.size(0), len(disc_true_table)
            pred_x = torch.zeros(B, total_feat, device=z.device)
            pred_x[:, disc_true_table] = recon_disc_x.float()
            # pred_x[:, ~disc_true_table] = recon_cont_x
            pred_x[:, ~disc_true_table] = recon_cont_x_final

            return [pred_x, recon_cont_x_final, recon_disc_x]

    def encode(self,x):
        # x: [B, 2*input_dim], 前半是masked data，后半是mask
        z_head=self.encoder_head.encode(x)
        z=self.encoder(z_head)
        return z


# --- 使用範例 ---
# 假設輸入維度為 784，隱藏層維度為 400，潛在維度為 20
# model = WAE(input_dim=784, hidden_dim=400, latent_dim=20)
#
# # 模擬輸入
# input_data = torch.randn(64, 784) # batch_size=64
#
# # 前向傳播
# recon_x, z = model(input_data)
#
# print("Reconstructed x shape:", recon_x.shape) # torch.Size([64, 784])
# print("Latent z shape:", z.shape)             # torch.Size([64, 20])