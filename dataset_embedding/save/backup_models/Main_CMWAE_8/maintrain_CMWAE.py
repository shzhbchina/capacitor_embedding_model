import torch
from torch.utils.data import DataLoader,random_split
from data.dataset import MyDataset
from train.BaseTrainer import BaseTrainer
from loss.MixedWAE_loss import MixedWAE_loss
from models.MT_CMixedWassersteinAutoEncoder import CMainMaskedMixedWAE
import joblib,os
from functools import partial
from utils.dataset_to_gpu import dataset_to_gpu


# 假设已有MyDataset、scaler等
file_abs_path = os.path.dirname(os.path.abspath(__file__))
#csv_path=os.path.join(file_abs_path,'data/datasheets/combine_large_xls','combined_large_excel_v4.2.csv')
# csv_path=os.path.join(file_abs_path,'data/datasheets/combine_large_xls','combined_large_excel_v4.3.csv')
#csv_path=os.path.join(file_abs_path,'data/datasheets/combine_large_xls','combined_large_excel_v4.3.1.csv')
csv_path=os.path.join(file_abs_path,'data/datasheets/combine_large_xls','combined_large_excel_v5.1.csv')
#scaler_save_path=os.path.join(file_abs_path,'save/scaler','scaler_params.save')
dataset = MyDataset(csv_path,mixed_transform=True)

#dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
processed_data_np=dataset.data_drop_fillna_log1cont_norm
true_table_df=dataset.true_table
gpu_dataset=dataset_to_gpu(processed_data_np,true_table_df)
dataloader = DataLoader(gpu_dataset, batch_size=128, shuffle=True)
print(f'processed data in {gpu_dataset.tensors[0].device}')

train_ratio = 0.9
val_ratio = 1-train_ratio
total_size = len(dataset)
train_size = int(total_size * train_ratio)
val_size = total_size - train_size
# 固定随机种子
generator = torch.Generator().manual_seed(42)
# 随机切分
#train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
train_set, val_set = random_split(gpu_dataset, [train_size, val_size], generator=generator)

device='cpu'#'cuda'
device = "cuda" if torch.cuda.is_available() else "cpu"



training_config={
    'num_epoch':1600,
    'device':device,
    'optimizer_name':'AdamW',
    'optimizer_lr':1e-4,
    'optimizer_decay_weight':0.001,
    'scheduler_name':'CosineAnnealingLR',
    'scheduler_params':{'CosineAnnealingLR':{'T_max': 2000,'eta_min':1e-7},
                        'StepLR':{'step_size': 30, 'gamma': 0.9},
                        'ReduceLRONPlateau':{'mode': 'min', 'patience': 10, 'factor': 0.9},
                        'OneCycleLR':{'max_lr': 0.001, 'total_steps': 100}},
    'gauss_noise':0.02,#0.01
    'batch_size':256,
    'save_name':'Main_CMWAE'
}
loss_conf={
    'reg_weight':5.0,
    'volt_weight':6,
    'cap_weight':6,
    'vol_weight':6,
    'price_weight':3,
    'ESR_weight':2,#0.5
    'ripple_weight':0.2,#0.5
}


disc_true_table=dataset.data_remain_columns.isin(dataset.category_column)
#discrete_feature_list={'true_table':[0,1,0,1...],'dimensions':{'cat_num':[2,10,2],'embedding_dim':[1,2,1]}}
discrete_feature_list={'true_table':disc_true_table,
                       'dimensions':{'cat_num':[2,10,2],'embedding_dim':[1,2+1,1]}}
encoder_path=os.path.join(file_abs_path,'save/Pre_MWAE/best_model.pth')
model=CMainMaskedMixedWAE(input_dim=dataset.shape[1],hidden_dim=32*8,latent_dim=8,disc_dim=4*8,discrete_feature_list=discrete_feature_list,dropout=0.01,encoder_path=encoder_path)
MixedWAE_loss_disc=partial(MixedWAE_loss,disc_table=disc_true_table,loss_conf=loss_conf)
trainer=BaseTrainer(model,MixedWAE_loss_disc,train_set,val_set,training_config,input_concate_true_table=True)


trainer.train()

