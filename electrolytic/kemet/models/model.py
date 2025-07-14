import torch
import torch.nn as nn

class MAE(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim=128):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, mask):
        # mask: bool, 1=keep, 0=mask
        masked_x = x.clone()
        masked_x[~mask] = 0
        emb = self.encoder(masked_x)
        recon = self.decoder(emb)
        return recon, emb
