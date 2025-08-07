import torch
import torch.nn as nn
import numpy as np
from dataset_embedding.models.PT_MixedWassersteinAutoEncoder import MaskedMixedWAE

class MainMaskedMixedWAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, disc_dim, discrete_feature_list, dropout=0.001,encoder_path=None):
        super(MainMaskedMixedWAE, self).__init__()
        self.discrete_feature_list = discrete_feature_list

        disc_true_table = discrete_feature_list['true_table']
        tot_discs = np.sum(disc_true_table.astype(int))
        tot_embedding_dims = np.sum(discrete_feature_list['dimensions']['embedding_dim']).astype(int)
        self.input_dim = input_dim  # 原始特征数

        eh_hidden_dim=32 * 4
        eh_latent_dim=16 * 2
        eh_disc_dim=4 * 4
        self.encoder_head=MaskedMixedWAE(input_dim, eh_hidden_dim, eh_latent_dim, eh_disc_dim, discrete_feature_list, dropout=0.0)
        if encoder_path is not None:
            self.encoder_head.load_state_dict(torch.load(encoder_path))
        for param in self.encoder_head.parameters():
            param.requires_grad = False

        # 其余结构和你第二阶段完全一样
        self.encoder = nn.Sequential(
            nn.Linear(eh_latent_dim, hidden_dim),  # 多一倍输入
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )


        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim - tot_discs)
        )

        embedding_dimensions = discrete_feature_list['dimensions']
        self.decoder_disc_layers = nn.ModuleList()
        for category_size in embedding_dimensions['cat_num']:
            layer = nn.Sequential(
                nn.Linear(latent_dim, disc_dim),
                nn.Dropout(dropout),
                nn.LayerNorm(disc_dim),
                nn.ReLU(),
                nn.Linear(disc_dim, category_size)
            )
            self.decoder_disc_layers.append(layer)

    def forward(self, x):
        # x: [B, 2*input_dim], 前半是masked data，后半是mask
        z=self.encode(x) # 拼接顺序: 连续 +embedding_disc + 连续mask  + 离散mask

        disc_true_table = self.discrete_feature_list['true_table']
        decode_vectors = []
        logits_vectors = []
        for i, layer in enumerate(self.decoder_disc_layers):
            disc_feature = layer(z)
            single_cat = torch.argmax(disc_feature, dim=1)
            logits_vectors.append(disc_feature)
            decode_vectors.append(single_cat)
        recon_disc_x = torch.stack(decode_vectors, dim=1)
        recon_cont_x = self.decoder(z)

        batch_size = z.shape[0]
        total_features = len(disc_true_table)
        pred_x = torch.zeros(batch_size, total_features, device=z.device)
        pred_x[:, disc_true_table] = recon_disc_x.float()
        pred_x[:, ~disc_true_table] = recon_cont_x

        return [[pred_x, recon_cont_x,logits_vectors], z]

    def decode(self, z):
        # 跟第二阶段完全一致
        disc_true_table = self.discrete_feature_list['true_table']
        with torch.no_grad():
            decode_vectors = []
            logits_vectors = []
            for i, layer in enumerate(self.decoder_disc_layers):
                disc_feature = layer(z)
                single_cat = torch.argmax(disc_feature, dim=1)
                logits_vectors.append(disc_feature)
                decode_vectors.append(single_cat)
            recon_disc_x = torch.stack(decode_vectors, dim=1)
            recon_cont_x = self.decoder(z)

            batch_size = z.shape[0]
            total_features = len(disc_true_table)
            pred_x = torch.zeros(batch_size, total_features, device=z.device)
            pred_x[:, disc_true_table] = recon_disc_x.float()
            pred_x[:, ~disc_true_table] = recon_cont_x
            return [pred_x, recon_cont_x, recon_disc_x]

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