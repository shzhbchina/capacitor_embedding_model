import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

class MyDataset(Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file (str): 数据文件路径
            transform (callable, optional): 预处理变换
        """
        self.data = pd.read_csv(csv_file)
        self.drop_cols = ['KEMET Part Number','Shape', 'Type', 'Manufacturer',
                          'VDC  Surge Voltage /V','DC Leakage /mA','MSL /Reflow Temp 260 deg',
                          'SPQ','MOQ','Impedance /mohm 20deg 10 kHz','ESL /nH','Dissipation Factor'
                          ]
        self.data_drop= self.data.drop(columns=self.drop_cols)
        self.col_mask = pd.Series([col not in self.drop_cols for col in self.data.columns],
                                  index=self.data.columns) #whether the column dropped
        self.data_drop_fillna=self.data_drop.fillna(0)
        self.scaler = StandardScaler()
        X_train=self.data_drop_fillna.values
        self.scaler.fit(X_train)
        self.shape=self.data_drop.shape


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data_drop_fillna.iloc[idx]
        true_table=(~self.data_drop.iloc[idx].isna()).astype('float32').values
        X = row.values.astype('float32')
        # 归一化（只特征归一化，不归一化y）
        X = self.scaler.transform([X])[0]
        X=X.astype('float32')
        return torch.tensor(X),torch.tensor(true_table)


# test code
# mydataset=MyDataset('datasheets/combine_large_xls/combined_large_excel_v3.1.csv')
# result=mydataset[2]
# original_value=mydataset.scaler.inverse_transform([result])
# original_value_test=mydataset.data_drop.iloc[2].values.astype('float32')
# print('test end')