import torch
from torch.utils.data import DataLoader,random_split
from data.dataset import MyDataset
from train.BaseTrainer import BaseTrainer
from loss.MixedWAE_loss import MixedWAE_loss
from models.MT_RCMixedWassersteinAutoEncoder import RCMainMaskedMixedWAE
import joblib,os
from functools import partial
from utils.dataset_to_gpu import dataset_to_gpu
import numpy as np
from sklearn.mixture import GaussianMixture
from dataset_embedding.utils.plot_func import plot_zdist
from dataset_embedding.utils.cal_GMM import cal_GMM,density_penalty_gmm


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

disc_true_table=dataset.data_remain_columns.isin(dataset.category_column)
#discrete_feature_list={'true_table':[0,1,0,1...],'dimensions':{'cat_num':[2,10,2],'embedding_dim':[1,2,1]}}
discrete_feature_list={'true_table':disc_true_table,
                       'dimensions':{'cat_num':[2,10,2],'embedding_dim':[1,2+1,1]}}
encoder_path=os.path.join(file_abs_path,'save/Pre_MWAE/best_model.pth')
model=RCMainMaskedMixedWAE(input_dim=dataset.shape[1],hidden_dim=32*8,latent_dim=8,disc_dim=4*8,
                           discrete_feature_list=discrete_feature_list,dropout=0.01,encoder_path=encoder_path)

save_name='Main_RCMWAE'
current_script_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(current_script_dir, 'save', save_name)
os.makedirs(save_dir + '', exist_ok=True)
best_model_path = os.path.join(save_dir, 'best_model.pth')
model.load_state_dict(torch.load(best_model_path,map_location=torch.device('cpu')))

x_train=np.concatenate((processed_data_np.values,true_table_df.values),axis=1)
z_train=model.encode(torch.tensor(x_train).float())
z_train=z_train.detach().numpy()

# latent_dim = z_train.shape[1]
# plt_data=z_train.detach().numpy()
# plot_zdist(plt_data=plt_data,dim=1)

gmm,z_mu,z_std=cal_GMM(z_train)

#test code
latent_dim=z_train.shape[1]
samples=100
seed_gen = torch.Generator().manual_seed(42)
z_test = torch.randn(samples, latent_dim, generator=seed_gen)
penalty=density_penalty_gmm(z_test, z_mu, z_std, gmm,clip=100.0)
print(penalty.mean(), np.percentile(penalty, [50,90,95])) #check random distance

pen_train = density_penalty_gmm(z_train, z_mu, z_std, gmm, clip=1e9)  # 不裁剪
print(np.percentile(pen_train, [50, 90, 95, 99])) #check training data distance
z_samp_std, _ = gmm.sample(1000)   # 在标准化坐标采样
z_samp = z_samp_std * z_std + z_mu # 还原到原 z 坐标
pen_samp = density_penalty_gmm(z_samp, z_mu, z_std, gmm, clip=1e9)
print(pen_samp.mean(), np.percentile(pen_samp, [50,90,95])) #  check gmm sample distance


# save code
save_dir_gmm=os.path.join(current_script_dir, 'save', 'gmm')
os.makedirs(save_dir_gmm + '', exist_ok=True)
save_path_gmm=os.path.join(save_dir_gmm, 'gmm.joblib')
save_part={'z_mu':z_mu,'z_std':z_std,'gmm':gmm}
joblib.dump(save_part, save_path_gmm)
pay_load=joblib.load(save_path_gmm)


