import torch
import torch.nn as nn
import numpy as np

class MaskedMixedWAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, disc_dim, discrete_feature_list, dropout=0.001):
        super(MaskedMixedWAE, self).__init__()
        self.discrete_feature_list = discrete_feature_list

        disc_true_table = discrete_feature_list['true_table']
        tot_discs = np.sum(disc_true_table.astype(int))
        tot_embedding_dims = np.sum(discrete_feature_list['dimensions']['embedding_dim']).astype(int)
        self.input_dim = input_dim  # 原始特征数

        # 其余结构和你第二阶段完全一样
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2 - tot_discs + tot_embedding_dims, hidden_dim),  # 多一倍输入
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

        self.embedding_layers = nn.ModuleList()
        embedding_dimensions = discrete_feature_list['dimensions']
        for num_categories, embedding_size in zip(embedding_dimensions['cat_num'], embedding_dimensions['embedding_dim']):
            self.embedding_layers.append(
                nn.Embedding(num_embeddings=num_categories, embedding_dim=embedding_size)
            )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            #nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim - tot_discs)
        )

        self.decoder_disc_layers = nn.ModuleList()
        for category_size in embedding_dimensions['cat_num']:
            layer = nn.Sequential(
                nn.Linear(latent_dim, disc_dim),
                nn.LayerNorm(disc_dim),
                nn.SiLU(),
                # nn.Dropout(dropout),
                nn.Linear(disc_dim, disc_dim),
                nn.LayerNorm(disc_dim),
                nn.SiLU(),
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
        B, D2 = x.shape
        assert D2 == 2 * self.input_dim, "输入shape应为 [B, 2*input_dim]"
        input_dim = self.input_dim

        masked_data = x[:, :input_dim]     # [B, input_dim]
        mask = x[:, input_dim:]            # [B, input_dim]
        disc_true_table = self.discrete_feature_list['true_table']

        # --- embedding离散 ---
        disc_idx = np.where(disc_true_table)[0]
        embedding_vectors = []
        for i, layer in enumerate(self.embedding_layers):
            input_column = masked_data[:, disc_idx[i]]  # 用掩码后的离散数据
            embedded_feature = layer(input_column.long())
            embedding_vectors.append(embedded_feature)
        embedding_disc = torch.cat(embedding_vectors, dim=1)  # [B, sum(embedding_dim)]

        # 连续特征和mask拼接
        x_cont = masked_data[:, ~disc_true_table]         # [B, num_cont]
        mask_cont = mask[:, ~disc_true_table]             # [B, num_cont]
        x_disc_emb = embedding_disc
        mask_disc = mask[:, disc_true_table]              # [B, num_disc]

        # 拼接顺序: 连续 +embedding_disc + 连续mask  + 离散mask
        x_cat = torch.cat([x_cont, x_disc_emb,mask_cont , mask_disc], dim=1)
        # Encoder输出
        z = self.encoder(x_cat)
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