from dataset_embedding.data.dataset import MyDataset
import joblib,os
from dataset_embedding.models.WassersteinAutoEncoder import WAE
import torch
import pandas as pd
import numpy as np
from query.query_datasheet import query_datasheet

#get class parameters, requires dataset+model
file_abs_path=os.path.dirname(os.path.abspath(__file__))
file_parent_path=os.path.dirname(file_abs_path)
datasheet_path=os.path.join(file_parent_path,'dataset_embedding/data/datasheets/combine_large_xls/combined_large_excel_v4.2.csv')
dataset=MyDataset(csv_file=datasheet_path)
model=WAE(input_dim=dataset.shape[1], hidden_dim=32, latent_dim=8)
model_name='WAE'
model_save_path=os.path.join(file_parent_path,'dataset_embedding/save',model_name,'best_model.pth')
model.load_state_dict(torch.load(model_save_path))

scaler_save_path=os.path.join(file_parent_path,'dataset_embedding/save','scaler/scaler_params.save')
test_query_datasheet=query_datasheet(model=model,dataset=dataset,scaler_save_path=scaler_save_path)

#test input
seed_gen=torch.Generator().manual_seed(123)
test_input=torch.randn(2,8,generator=seed_gen)
test_output=test_query_datasheet.find_component_from_latent_space(test_input)

test_input_dataset,_=dataset[0]
test_output_dataset_x,test_output_dataset_z=model(test_input_dataset)
test_output_dataset=test_query_datasheet.find_component_from_latent_space(test_output_dataset_z.unsqueeze(0))