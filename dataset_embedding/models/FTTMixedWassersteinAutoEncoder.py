import torch
import torch.nn as nn
import numpy as np






class FeatureTokenizer(nn.Module):
    def __init__(self, num_continuous, categorical_info, embed_dim):
        super().__init__()
        self.num_continuous = num_continuous
        self.num_categorical = len(categorical_info['cat_num'])

        # 為每個分類特徵建立 Embedding 層
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(num_classes, embed_dim)
            for num_classes in categorical_info['cat_num']
        ])

        # 為連續特徵建立一個共享的線性投射層和偏置項
        if num_continuous > 0:
            self.cont_projection = nn.Linear(1, embed_dim, bias=False)
            self.cont_biases = nn.Parameter(torch.zeros(num_continuous, embed_dim))

    def forward(self, x_cont, x_cat):
        tokens = []

        # 處理分類特徵
        cat_embeds = [
            embed(x_cat[:, i].unsqueeze(1)) for i, embed in enumerate(self.embedding_layers)
        ]
        tokens.extend(cat_embeds)

        # 處理連續特徵
        if self.num_continuous > 0:
            # [batch_size, num_cont] -> [batch_size, num_cont, 1]
            x_cont = x_cont.unsqueeze(-1)
            # [batch_size, num_cont, 1] -> [batch_size, num_cont, embed_dim]
            cont_embeds = self.cont_projection(x_cont) + self.cont_biases
            tokens.append(cont_embeds)

        # 將所有 token 拼接起來
        # 最終形狀: [batch_size, num_features, embed_dim]
        return torch.cat(tokens, dim=1)


# --- 步驟二：建立 FT-Transformer Autoencoder 主模型 ---
class FTTMixedWAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, discrete_feature_list,
                 nhead=8, num_encoder_layers=2, dropout=0.1):
        super().__init__()
        self.discrete_feature_list = discrete_feature_list
        disc_true_table = discrete_feature_list['true_table']

        # 計算連續和分類特徵的數量
        self.num_categorical_features = np.sum(disc_true_table.astype(int))
        self.num_continuous_features = input_dim - self.num_categorical_features

        # Transformer 的嵌入維度，hidden_dim 現在作為 d_model
        embed_dim = hidden_dim

        # --- Encoder 部分 ---
        # 1. Feature Tokenizer
        self.feature_tokenizer = FeatureTokenizer(
            num_continuous=self.num_continuous_features,
            categorical_info=discrete_feature_list['dimensions'],
            embed_dim=embed_dim
        )

        # 2. CLS Token (Class Token)
        # 這是我們最終要用來代表整個樣本的潛在向量 z 的前身
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True  # 確保輸入形狀是 [batch_size, seq_len, embed_dim]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 4. 從 CLS Token 輸出到最終的潛在向量 z
        #self.latent_head = nn.Linear(embed_dim, latent_dim)
        self.latent_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, latent_dim)
        )

        # --- Decoder 部分 (與您之前的設計類似，但更簡化) ---
        self.decoder_mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Decoder 的多頭輸出
        self.decoder_cont_head = nn.Linear(hidden_dim, self.num_continuous_features)
        self.decoder_disc_heads = nn.ModuleList()
        for category_size in discrete_feature_list['dimensions']['cat_num']:
            self.decoder_disc_heads.append(nn.Linear(hidden_dim, category_size))

    def forward(self, x):
        # 準備數據
        disc_true_table = self.discrete_feature_list['true_table']
        x_cont = x[:, ~disc_true_table]
        x_cat = x[:, disc_true_table].long()  # Embedding 層需要 LongTensor

        # --- Encoding ---
        # 1. 將所有特徵轉換為 token
        tokens = self.feature_tokenizer(x_cont, x_cat)  # Shape: [B, num_features, D]

        # 2. 在最前面加上 CLS token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        tokens_with_cls = torch.cat([cls_tokens, tokens], dim=1)  # Shape: [B, num_features + 1, D]

        # 3. 通過 Transformer Encoder
        encoded_tokens = self.transformer_encoder(tokens_with_cls)

        # 4. 提取 CLS token 的輸出，並投射到潛在空間 z
        cls_output = encoded_tokens[:, 0]  # 取出第一個 token ([CLS])
        z = self.latent_head(cls_output)  # Shape: [B, latent_dim]


        # --- Decoding ---
        # 使用潛在向量 z 來重建原始數據
        hidden_rep = self.decoder_mlp(z)
        recon_cont_x = self.decoder_cont_head(hidden_rep)
        recon_disc_logits = [head(hidden_rep) for head in self.decoder_disc_heads]
        recon_disc_labels = torch.stack([torch.argmax(logits, dim=1) for logits in recon_disc_logits], dim=1)

        batch_size = z.shape[0]
        disc_true_table = self.discrete_feature_list['true_table']
        total_features = len(disc_true_table)
        device = z.device

        pred_x = torch.zeros(batch_size, total_features, device=device)
        pred_x[:, disc_true_table] = recon_disc_labels.float()
        pred_x[:, ~disc_true_table] = recon_cont_x


        return [[pred_x, recon_cont_x, recon_disc_logits],z]

    def decode(self, z):
        with torch.no_grad():
            # 使用潛在向量 z 來重建原始數據
            hidden_rep = self.decoder_mlp(z)

            # 重建連續部分
            recon_cont_x = self.decoder_cont_head(hidden_rep)

            # 重建離散部分的 logits
            recon_disc_logits = [head(hidden_rep) for head in self.decoder_disc_heads]

            # --- 組裝完整輸出 (這部分可以放在 Trainer 的 loss function 中處理) ---
            # 為了保持接口一致，我們在這裡組裝
            recon_disc_labels = torch.stack([torch.argmax(logits, dim=1) for logits in recon_disc_logits], dim=1)

            batch_size = z.shape[0]
            disc_true_table = self.discrete_feature_list['true_table']
            total_features = len(disc_true_table)
            device = z.device

            pred_x = torch.zeros(batch_size, total_features, device=device)
            pred_x[:, disc_true_table] = recon_disc_labels.float()
            pred_x[:, ~disc_true_table] = recon_cont_x

            return [pred_x, recon_cont_x, recon_disc_labels.float()]


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