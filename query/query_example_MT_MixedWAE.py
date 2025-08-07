from torch.utils.data import DataLoader

from dataset_embedding.data.dataset import MyDataset
import joblib,os
#from dataset_embedding.models.WassersteinAutoEncoder import WAE
from dataset_embedding.models.MT_MixedWassersteinAutoEncoder import MainMaskedMixedWAE
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
from dataset_embedding.utils.scatter_3d_slice import scatter_3d_slice
from dataset_embedding.utils.curve_selected_dimension import curve_selected_dimension

#get class parameters, requires dataset+model
file_abs_path=os.path.dirname(os.path.abspath(__file__))
file_parent_path=os.path.dirname(file_abs_path)
#datasheet_path=os.path.join(file_parent_path,'dataset_embedding/data/datasheets/combine_large_xls/combined_large_excel_v4.2.csv')
datasheet_path=os.path.join(file_parent_path,'dataset_embedding/data/datasheets/combine_large_xls/combined_large_excel_v4.3.csv')
dataset=MyDataset(csv_file=datasheet_path,mixed_transform=True)
disc_true_table=dataset.data_remain_columns.isin(dataset.category_column)
discrete_feature_list={'true_table':disc_true_table,
                       'dimensions':{'cat_num':[2,10,2],'embedding_dim':[1,2+1,1]}}
hidden_dim=32*8
latent_dim=16
encoder_path=os.path.join(file_parent_path,'dataset_embedding/save/Pre_MWAE/best_model.pth')
model=MainMaskedMixedWAE(input_dim=dataset.shape[1], hidden_dim=hidden_dim, latent_dim=latent_dim,disc_dim=4*8,discrete_feature_list=discrete_feature_list,encoder_path=encoder_path)
model_name='Main_MWAE'
model_save_path=os.path.join(file_parent_path,'dataset_embedding/save',model_name,'best_model.pth')
model.load_state_dict(torch.load(model_save_path, map_location='cpu'))

# scaler_save_path=os.path.join(file_parent_path,'dataset_embedding/save','scaler/scaler_params.save')
test_query_datasheet=query_datasheet(model=model,dataset=dataset,scaler_save_path=None)

#test input
seed_gen=torch.Generator().manual_seed(123)
extreme_amp=1
test_input=torch.randn(2,latent_dim,generator=seed_gen)+extreme_amp
test_output=test_query_datasheet.find_component_from_latent_space(test_input,mixed_transform=True)

shuffled=True
if not shuffled:
    test_input_dataset,test_true_table=dataset[0:1000]
    test_output_dataset_x,test_output_dataset_z=model(torch.cat((test_input_dataset,test_true_table),dim=1))
    with torch.no_grad():
        test_output_dataset=test_query_datasheet.find_component_from_latent_space(test_output_dataset_z,mixed_transform=True)
    test_input_dataset_original=dataset.data_drop_fillna.iloc[0:1000]
else:
    N = len(dataset)
    sample_num = 1000
    idx = np.random.choice(N, size=sample_num, replace=False)
    test_input_dataset, test_true_table = dataset[idx]
    test_output_dataset_x, test_output_dataset_z = model(torch.cat((test_input_dataset,test_true_table),dim=1))
    with torch.no_grad():
        test_output_dataset = test_query_datasheet.find_component_from_latent_space(test_output_dataset_z,
                                                                                    mixed_transform=True)
    test_input_dataset_original = dataset.data_drop_fillna.iloc[idx]

#error
#[loss, recon_loss, mmd_loss] = MixedWAE_loss(test_output_dataset_x,test_input_dataset,test_output_dataset_z,test_true_table,disc_true_table)
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

#  plot z
latent_dim = test_output_dataset_z.shape[1]
plt_data=test_output_dataset_z.detach().numpy()
plot_zdist(plt_data=plt_data,dim=1)

#plot dataset 3d
plot_dataset=dataset.data_drop
plot_dataset['Volume'] = np.where(
    plot_dataset['Shape_code'] == 0,
    plot_dataset['Diameter /mm'] ** 2 * plot_dataset['Cylinder_length /mm'],
    plot_dataset['Length /mm'] * plot_dataset['Width /mm'] * plot_dataset['Height /mm']
)
scatter_3d_slice(plot_dataset, "Rated Voltage /V", "Rated Capacitance /uF","Volume" )
# 举例：只筛选体积在100附近的数据
test=scatter_3d_slice(plot_dataset, "Rated Voltage /V",  "Rated Capacitance /uF","Volume", highlight_col="Volume", highlight_value=10000)

#
seed_gen = torch.Generator().manual_seed(42)
z_fixed = torch.randn(1, latent_dim, generator=seed_gen)
base_z=z_fixed
z_dim=0
show_column_name=['Rated Voltage /V']
curve_selected_dimension(test_query_datasheet,base_z,z_dim,show_column_name,samples=100)