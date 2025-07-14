import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

class MyDataset(Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file (str): 数据文件路径
            transform (callable, optional): 预处理变换
        """
        self.data = pd.read_csv(csv_file)
        self.drop_cols = ['KEMET Part Number', 'Shape', 'Type', 'Manufacturer']
        self.data_drop= self.data.drop(columns=self.drop_cols)
        self.scaler = StandardScaler()
        X_train=self.data_drop.values
        self.scaler.fit(X_train)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data_drop.iloc[idx]
        X = row.values.astype('float32')
        # 归一化（只特征归一化，不归一化y）
        X = self.scaler.transform([X])[0]
        return torch.tensor(X)


# test code
# mydataset=MyDataset('datasheets/combine_large_xls/combined_large_excel_v3.1.csv')
# result=mydataset[2]
# original_value=mydataset.scaler.inverse_transform([result])
# original_value_test=mydataset.data_drop.iloc[2].values.astype('float32')
# print('test end')