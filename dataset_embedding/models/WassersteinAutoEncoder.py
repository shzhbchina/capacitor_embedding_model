import torch
import torch.nn as nn


class WAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(WAE, self).__init__()

        # WAE 的編碼器：直接輸出一個潛在向量 z
        # 不再需要 mu 和 logvar 兩個頭
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)  # 直接輸出 z
        )

        # WAE 的解碼器：結構可以與 VAE 保持一致
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )


    def forward(self, x):
        # 1. 編碼：將輸入 x 確定性地映射到潛在向量 z
        z = self.encoder(x)

        # 2. 解碼：從潛在向量 z 重建回 x'
        recon_x = self.decoder(z)

        # 3. 返回重建結果和潛在向量 z，以供 WAE 損失函數使用
        # WAE loss 需要 z 來計算 MMD 損失
        return [recon_x, z]

    def decode(self, z):
        return self.decoder(z)

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