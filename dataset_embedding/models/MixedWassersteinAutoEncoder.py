import torch
import torch.nn as nn
import numpy as np



class MixedWAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim,disc_dim,discrete_feature_list,dropout=0.001):
        #discrete_feature_list={'true_table':[0,1,0,1...],'dimensions':{'cat_num':[2,10,2],'embedding_dim':[1,2,1]}}
        super(MixedWAE, self).__init__()
        self.discrete_feature_list=discrete_feature_list

        disc_true_table=discrete_feature_list['true_table']
        tot_discs=np.sum(disc_true_table.astype(int))
        tot_embedding_dims=np.sum(discrete_feature_list['dimensions']['embedding_dim']).astype(int)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim-tot_discs+tot_embedding_dims, hidden_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)  # 直接輸出 z
        )

        self.embedding_layers = nn.ModuleList()
        embedding_dimensions=discrete_feature_list['dimensions']
        for num_categories, embedding_size in zip(embedding_dimensions['cat_num'], embedding_dimensions['embedding_dim']):
            layer = nn.Embedding(num_embeddings=num_categories, embedding_dim=embedding_size)
            self.embedding_layers.append(layer)


        # WAE 的解碼器：結構可以與 VAE 保持一致
        tot_embeddings=np.sum(discrete_feature_list['dimensions']['embedding_dim'])
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim-tot_discs),
        )

        self.decoder_disc_layers=nn.ModuleList()
        embedding_dimensions=discrete_feature_list['dimensions']
        for category_size in embedding_dimensions['cat_num']:
            layer = nn.Sequential(
                nn.Linear(latent_dim, disc_dim),
                nn.Dropout(dropout),
                nn.LayerNorm(disc_dim),
                nn.ReLU(),
                nn.Linear(disc_dim, disc_dim ),
                nn.Dropout(dropout),
                nn.LayerNorm(disc_dim),
                nn.ReLU(),
                nn.Linear(disc_dim , disc_dim),
                nn.Dropout(dropout),
                nn.LayerNorm(disc_dim),
                nn.ReLU(),
                nn.Linear(disc_dim,category_size),
            )
            self.decoder_disc_layers.append(layer)


    def forward(self, x):
        if x.dim()==1:
            x=x.unsqueeze(0)

        disc_true_table=self.discrete_feature_list['true_table']

        disc_idx=np.where(disc_true_table)[0]
        embedding_vectors=[]
        for i, layer in enumerate(self.embedding_layers):
            input_column = x[:, disc_idx[i]]
            embedded_feature = layer(input_column.long())
            embedding_vectors.append(embedded_feature)
        embedding_disc=torch.cat(embedding_vectors,dim=1)
        x_cat = torch.cat([x[:,~disc_true_table], embedding_disc], dim=1)
        z = self.encoder(x_cat)

        recon_x = self.decoder(z)
        decode_vectors=[]
        logits_vectors=[]
        for i,layer in enumerate(self.decoder_disc_layers):
            disc_feature=layer(z)
            single_cat=torch.argmax(disc_feature, dim=1)
            logits_vectors.append(disc_feature)
            decode_vectors.append(single_cat)
        recon_disc_x=torch.stack(decode_vectors,dim=1)
        recon_cont_x = self.decoder(z)

        batch_size = z.shape[0]
        total_features = len(disc_true_table)
        pred_x = torch.zeros(batch_size, total_features, device=z.device)
        pred_x[:,disc_true_table]=recon_disc_x.float()
        pred_x[:, ~disc_true_table] = recon_cont_x

        return [[pred_x, recon_cont_x, logits_vectors], z]

    def decode(self, z):
        disc_true_table = self.discrete_feature_list['true_table']
        with torch.no_grad():

            recon_x = self.decoder(z)
            decode_vectors=[]
            logits_vectors=[]
            for i,layer in enumerate(self.decoder_disc_layers):
                disc_feature=layer(z)
                single_cat=torch.argmax(disc_feature, dim=1)
                logits_vectors.append(disc_feature)
                decode_vectors.append(single_cat)
            recon_disc_x=torch.stack(decode_vectors,dim=1)
            recon_cont_x = self.decoder(z)

            batch_size = z.shape[0]
            total_features = len(disc_true_table)
            pred_x = torch.zeros(batch_size, total_features, device=z.device)
            pred_x[:,disc_true_table]=recon_disc_x.float()
            pred_x[:, ~disc_true_table] = recon_cont_x
            return  [pred_x, recon_cont_x, recon_disc_x]

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