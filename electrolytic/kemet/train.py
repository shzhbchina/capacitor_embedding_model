import torch
from torch.utils.data import DataLoader,random_split
from models.model import MAE
from loss.loss import mae_loss
from data.dataset import MyDataset

# 假设已有MyDataset、scaler等
csv_path=r'data/datasheets/combine_large_xls/combined_large_excel_v3.1.csv'
dataset = MyDataset(csv_path)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)


train_ratio = 0.9
val_ratio = 1-train_ratio
total_size = len(dataset)
train_size = int(total_size * train_ratio)
val_size = total_size - train_size
# 固定随机种子
generator = torch.Generator().manual_seed(42)
# 随机切分
train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
# DataLoader
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
val_loader = DataLoader(val_set, batch_size=128, shuffle=False)

device='cpu'#'cuda'

model = MAE(input_dim=dataset.data_drop.shape[1], embedding_dim=16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    for X in dataloader:
        X = X.to(device)
        # 随机掩码生成
        mask = (torch.rand_like(X) > 0.3)
        recon, emb = model(X, mask)
        loss = mae_loss(recon, X, mask, emb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch}, Loss: {loss.item():.6f}')
