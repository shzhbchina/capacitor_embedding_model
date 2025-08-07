import torch
import torch.nn as nn
import numpy as np

# ------- Transformer Encoder for Tabular Data -------
class TabAttentionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_heads=4, n_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.token_emb = nn.Linear(1, hidden_dim)   # 每个特征独立投影为token embedding
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x_cat):
        # x_cat: (batch, feature_dim)
        x = x_cat.unsqueeze(-1)                   # (B, F, 1)
        x = self.token_emb(x)                     # (B, F, H)
        x = self.transformer(x)                   # (B, F, H)
        pooled = x.mean(dim=1)                    # (B, H)
        out = self.output_proj(pooled)            # (B, latent_dim)
        return out

# ------- Transformer Decoder for Tabular Data -------
class TabTransformerDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, out_dim, n_heads=4, n_layers=2):
        super().__init__()
        self.out_dim = out_dim
        # learnable token embedding for each output variable
        self.token_embed = nn.Parameter(torch.zeros(1, out_dim, hidden_dim))
        nn.init.xavier_uniform_(self.token_embed)
        self.z_proj = nn.Linear(latent_dim, hidden_dim)
        decoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*2, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(hidden_dim, 1)

    def forward(self, z):
        # z: (B, latent_dim)
        B = z.size(0)
        x = self.token_embed.expand(B, self.out_dim, -1)   # (B, out_dim, H)
        z_cond = self.z_proj(z).unsqueeze(1)               # (B, 1, H)
        x = x + z_cond                                     # (B, out_dim, H)
        x = self.decoder(x)                                # (B, out_dim, H)
        out = self.out_proj(x).squeeze(-1)                 # (B, out_dim)
        return out

# ------- MixedWAE主模型（接口完全不变，直接替换） -------
class TMWAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, disc_dim, discrete_feature_list):
        super(TMWAE, self).__init__()
        self.discrete_feature_list = discrete_feature_list
        disc_true_table = discrete_feature_list['true_table']
        tot_discs = np.sum(disc_true_table.astype(int))
        tot_embedding_dims = np.sum(discrete_feature_list['dimensions']['embedding_dim']).astype(int)

        # 编码器
        self.encoder = TabAttentionEncoder(
            input_dim=input_dim - tot_discs + tot_embedding_dims,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_heads=4,
            n_layers=2
        )

        self.embedding_layers = nn.ModuleList()
        embedding_dimensions = discrete_feature_list['dimensions']
        for num_categories, embedding_size in zip(embedding_dimensions['cat_num'], embedding_dimensions['embedding_dim']):
            layer = nn.Embedding(num_embeddings=num_categories, embedding_dim=embedding_size)
            self.embedding_layers.append(layer)

        # 解码器
        self.decoder = TabTransformerDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            out_dim=input_dim - tot_discs,
            n_heads=4,
            n_layers=2
        )

        self.decoder_disc_layers = nn.ModuleList()
        for category_size in embedding_dimensions['cat_num']:
            layer = nn.Sequential(
                nn.Linear(latent_dim, disc_dim),
                nn.LayerNorm(disc_dim),
                nn.ReLU(),
                nn.Linear(disc_dim, category_size),
            )
            self.decoder_disc_layers.append(layer)

    def forward(self, x):
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
        z = self.encoder(x_cat)

        recon_x = self.decoder(z)
        decode_vectors = []
        logits_vectors = []
        for i, layer in enumerate(self.decoder_disc_layers):
            disc_feature = layer(z)
            single_cat = torch.argmax(disc_feature, dim=1)
            logits_vectors.append(disc_feature)
            decode_vectors.append(single_cat)
        recon_disc_x = torch.stack(decode_vectors, dim=1)
        recon_cont_x = recon_x

        batch_size = z.shape[0]
        total_features = len(disc_true_table)
        pred_x = torch.zeros(batch_size, total_features, device=z.device)
        pred_x[:, disc_true_table] = recon_disc_x.float()
        pred_x[:, ~disc_true_table] = recon_cont_x

        return [[pred_x, recon_cont_x, logits_vectors], z]

    def decode(self, z):
        disc_true_table = self.discrete_feature_list['true_table']
        with torch.no_grad():
            recon_x = self.decoder(z)
            decode_vectors = []
            logits_vectors = []
            for i, layer in enumerate(self.decoder_disc_layers):
                disc_feature = layer(z)
                single_cat = torch.argmax(disc_feature, dim=1)
                logits_vectors.append(disc_feature)
                decode_vectors.append(single_cat)
            recon_disc_x = torch.stack(decode_vectors, dim=1)
            recon_cont_x = recon_x

            batch_size = z.shape[0]
            total_features = len(disc_true_table)
            pred_x = torch.zeros(batch_size, total_features, device=z.device)
            pred_x[:, disc_true_table] = recon_disc_x.float()
            pred_x[:, ~disc_true_table] = recon_cont_x
            return [pred_x, recon_cont_x, recon_disc_x]

