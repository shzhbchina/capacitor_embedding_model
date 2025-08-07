from torch.utils.data import DataLoader

from dataset_embedding.data.dataset import MyDataset
import joblib,os
from dataset_embedding.models.FTT2MixedWassersteinAutoEncoder import FTT2MixedWAE
import torch
import pandas as pd
import numpy as np
from query.query_datasheet import query_datasheet
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from dataset_embedding.utils.plot_func import plot_zdist
from dataset_embedding.loss.MixedWAE_loss import MixedWAE_loss

#get class parameters, requires dataset+model
file_abs_path=os.path.dirname(os.path.abspath(__file__))
file_parent_path=os.path.dirname(file_abs_path)
#datasheet_path=os.path.join(file_parent_path,'dataset_embedding/data/datasheets/combine_large_xls/combined_large_excel_v4.2.csv')
datasheet_path=os.path.join(file_parent_path,'dataset_embedding/data/datasheets/combine_large_xls/combined_large_excel_v4.3.csv')
dataset=MyDataset(csv_file=datasheet_path,mixed_transform=True)
disc_true_table=dataset.data_remain_columns.isin(dataset.category_column)
discrete_feature_list={'true_table':disc_true_table,
                       'dimensions':{'cat_num':[2,10,2],'embedding_dim':[1,2+1,1]}}
hidden_dim=32*1
latent_dim=16*1
model=FTT2MixedWAE(input_dim=dataset.shape[1], hidden_dim=hidden_dim, latent_dim=latent_dim,discrete_feature_list=discrete_feature_list,dropout=0.1)
model_name='FTT2MWAE'
model_save_path=os.path.join(file_parent_path,'dataset_embedding/save',model_name,'best_model.pth')
model.load_state_dict(torch.load(model_save_path, map_location='cpu'))

# scaler_save_path=os.path.join(file_parent_path,'dataset_embedding/save','scaler/scaler_params.save')
test_query_datasheet=query_datasheet(model=model,dataset=dataset,scaler_save_path=None)

#test input
seed_gen=torch.Generator().manual_seed(123)
test_input=torch.randn(2,latent_dim,generator=seed_gen)
with torch.no_grad():
    test_output=test_query_datasheet.find_component_from_latent_space(test_input,mixed_transform=True)

shuffled=True
if not shuffled:
    start_idx=0
    end_idx=1000
    test_input_dataset,test_true_table=dataset[start_idx:end_idx]
    test_output_dataset_x,test_output_dataset_z=model(test_input_dataset)
    with torch.no_grad():
        test_output_dataset=test_query_datasheet.find_component_from_latent_space(test_output_dataset_z,mixed_transform=True)
    test_input_dataset_original=dataset.data_drop_fillna.iloc[start_idx:end_idx]
else:
    N = len(dataset)
    sample_num = 1000
    idx = np.random.choice(N, size=sample_num, replace=False)
    test_input_dataset, test_true_table= dataset[idx]
    test_output_dataset_x,test_output_dataset_z=model(test_input_dataset)
    with torch.no_grad():
        test_output_dataset=test_query_datasheet.find_component_from_latent_space(test_output_dataset_z,mixed_transform=True)
    test_input_dataset_original=dataset.data_drop_fillna.iloc[idx]


loss_conf={
    'reg_weight':10.0,
    'volt_weight':6,
    'cap_weight':6,
    'price_weight':6,
    'ESR_ripple_weight':0.5,
}
#error
[loss, recon_loss, mmd_loss] = MixedWAE_loss(test_output_dataset_x,test_input_dataset,test_output_dataset_z,test_true_table,disc_true_table,loss_conf=loss_conf)
test_err=test_output_dataset_x[0]-test_input_dataset
print(torch.mean(torch.abs(test_err)))
#[loss, recon_loss, mmd_loss] = MixedWAE_loss(test_output_dataset_x,test_input_dataset,torch.randn_like(test_output_dataset_z),test_true_table,disc_true_table,loss_conf=loss_conf)
print(f'Loss:{loss}, Recon Loss:{recon_loss}, MMD Loss:{mmd_loss}')
#  plot
latent_dim = test_output_dataset_z.shape[1]
plt_data=test_output_dataset_z.detach().numpy()
#plt_data=torch.randn_like(test_output_dataset_z)
plot_zdist(plt_data=plt_data,dim=1)


