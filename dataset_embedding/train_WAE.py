import torch
from torch.utils.data import DataLoader,random_split
from data.dataset import MyDataset
from train.BaseTrainer import BaseTrainer
from loss.WAE_loss import WAE_loss
from models.WassersteinAutoEncoder import WAE
import joblib,os


# 假设已有MyDataset、scaler等
file_abs_path = os.path.dirname(os.path.abspath(__file__))
csv_path=os.path.join(file_abs_path,'data/datasheets/combine_large_xls','combined_large_excel_v4.2.csv')
scaler_save_path=os.path.join(file_abs_path,'save/scaler','scaler_params.save')
dataset = MyDataset(csv_path)
dataset.save_scaler_params(scaler_save_path)
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

device='cpu'#'cuda'
device = "cuda" if torch.cuda.is_available() else "cpu"



training_config={
    'num_epoch':800,
    'device':'cpu',
    'optimizer_name':'AdamW',
    'optimizer_lr':1e-4,
    'optimizer_decay_weight':0.001,
    'scheduler_name':'CosineAnnealingLR',
    'scheduler_params':{'CosineAnnealingLR':{'T_max': 1000,'eta_min':1e-7},
                        'StepLR':{'step_size': 30, 'gamma': 0.9},
                        'ReduceLRONPlateau':{'mode': 'min', 'patience': 10, 'factor': 0.9},
                        'OneCycleLR':{'max_lr': 0.001, 'total_steps': 100}},
    'save_name':'WAE'
}
model=WAE(input_dim=dataset.shape[1],hidden_dim=32,latent_dim=8)
trainer=BaseTrainer(model,WAE_loss,train_set,val_set,training_config)


trainer.train()

