import torch
from torch.utils.data import DataLoader,random_split
from models.MaskedAutoEncoder import MAE
from models.WassersteinAutoEncoder import WAE
from models.VariationalAutoEncoder import VAE
from loss.loss import mae_loss
from loss.VAE_loss import VAE_loss
from loss.WAE_loss import WAE_loss
from data.dataset import MyDataset
import os
from torch.utils.tensorboard import SummaryWriter

# 假设已有MyDataset、scaler等
csv_path=r'data/datasheets/combine_large_xls/combined_large_excel_v4.1.csv'
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
valid_loader = DataLoader(val_set, batch_size=128, shuffle=False)

device='cpu'#'cuda'
device = "cuda" if torch.cuda.is_available() else "cpu"




num_epochs = 100

def_train_VAE=0
def_train_WAE=1
if def_train_VAE:
    input_dim, hidden_dim, latent_dim = dataset.shape[1], 32, 16
    model = VAE(input_dim, hidden_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(100):
        loss_tot=0
        for data,true_table in train_loader:  # data应为归一化后的tensor
            [recon_batch, mu, logvar] = model(data)
            [loss,BCE,KLD] = VAE_loss([recon_batch, mu, logvar], data, true_table)
            optimizer.zero_grad()
            loss.backward()
            loss_tot+=loss.item()
            optimizer.step()
        print(f'Epoch {epoch}, Train Loss: {loss_tot/len(train_loader.dataset):.6f}',end='')

        loss_tot=0
        for data,true_table in valid_loader:  # data应为归一化后的tensor
            with torch.no_grad():
                [recon_batch, mu, logvar] = model(data)
                [loss,BCE,KLD] = VAE_loss([recon_batch, mu, logvar], data,true_table)
                loss_tot+=loss.item()
        print(f' Valid Loss: {loss_tot/len(valid_loader.dataset):.6f}')

if def_train_WAE:
    num_epoch=100
    input_dim, hidden_dim, latent_dim = dataset.shape[1], 32, 16
    model = WAE(input_dim, hidden_dim, latent_dim).to(device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,  # 您的學習率
        weight_decay=1e-2/10  # 一個需要調整的權重衰減值
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    #scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)#classical *gamma every setp_size
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1) #if not improve after patience
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, total_steps=100) #warmup
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(current_script_dir + '', exist_ok=True)
    save_dir = os.path.join(current_script_dir, 'save','WAE')
    checkpoint_path = os.path.join(save_dir, 'wae_checkpoint.pth')
    best_model_path = os.path.join(save_dir, 'best_wae_model.pth')
    writer = SummaryWriter(os.path.join(save_dir,'wae_experiment_1'))


    best_valid_loss = float('inf')  # 初始化一個無限大的最佳損失
    start_epoch=0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_valid_loss = checkpoint.get('best_valid_loss', float('inf'))  # .get 提供向下相容性
        print(f"成功加載檢查點，將從 Epoch {start_epoch} 開始訓練。")

    for epoch in range(start_epoch, num_epochs):
        loss_tot=0
        for data,true_table in train_loader:
            data=data.to(device)
            true_table=true_table.to(device)
            [recon_batch, pred_z] = model(data)
            [loss,recon_loss,mmd_loss] = WAE_loss(recon_batch,data,pred_z,true_table)
            optimizer.zero_grad()
            loss.backward()
            loss_tot+=loss.item()
            optimizer.step()
        loss_train=loss_tot/len(train_loader.dataset)
        print(f'Epoch {epoch}, Train Loss: {loss_train:.6f}',end='')
        writer.add_scalar('Loss/train', loss_train, epoch)

        loss_tot=0
        model.eval()
        for data,true_table in valid_loader:  # data应为归一化后的tensor
            with torch.no_grad():
                data = data.to(device)
                true_table = true_table.to(device)
                [recon_batch, pred_z] = model(data)
                [loss, recon_loss, mmd_loss] = WAE_loss(recon_batch, data, pred_z,true_table)
                loss_tot+=loss.item()
        valid_loss=loss_tot / len(valid_loader.dataset)
        print(f' Valid Loss: {valid_loss:.6f}')
        writer.add_scalar('Loss/train', valid_loss, epoch)
        scheduler.step()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_valid_loss': best_valid_loss
        }, checkpoint_path)

        # 如果當前的驗證損失比之前的最佳損失還要低，就儲存為最佳模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), best_model_path)
            #print(f'已儲存新的最佳模型，驗證損失為: {best_valid_loss:.6f}')
        writer.close()
# in terminal open tensorboard
#tensorboard --logdir=electrolytic/kemet/save/WAE
print('end')