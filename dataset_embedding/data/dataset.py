import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib,os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer


class MyDataset(Dataset):
    def __init__(self, csv_file,mixed_transform=False):
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
        self.data_remain_columns=self.data_drop.columns
        self.true_table = (~self.data_drop.isna())
        self.col_mask = pd.Series([col not in self.drop_cols for col in self.data.columns],
                                  index=self.data.columns) #whether the column dropped
        self.data_drop_fillna_raw=self.data_drop.fillna(0.001)
        self.data_drop_fillna = self.data_drop_fillna_raw.replace(0, 0.001)
        self.data_drop_fillna_log=np.log(self.data_drop_fillna)
        self.scaler = StandardScaler()
        X_train=self.data_drop_fillna_log.values
        self.scaler.fit(X_train)
        self.shape=self.data_drop.shape
        self.category_column=['Shape_code','Type_code','Manufacturer_code']

        self.continuous_column=[col for col in self.data_drop_fillna.columns if col not in self.category_column]
        log_transformer = FunctionTransformer(np.log1p, inverse_func=np.expm1)
        # 建立一個包含 log 和 scale 的處理管線
        continuous_pipeline = Pipeline([
            ('log', log_transformer),
            ('scaler', StandardScaler())
        ])
        # 建立 ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('cont', continuous_pipeline, self.continuous_column)
            ],
            remainder='passthrough'
        )
        processed_data = preprocessor.fit_transform(self.data_drop_fillna_raw)
        # ColumnTransformer 輸出時，passthrough 的欄位會被放到後面，我們需要還原順序
        restored_df = pd.DataFrame(processed_data, columns=self.continuous_column + self.category_column)
        self.data_drop_fillna_log1cont_norm = restored_df[self.data_drop_fillna_raw.columns]
        self.preprocessor=preprocessor


        # fitted_pipeline = self.preprocessor.named_transformers_['cont']
        # fitted_scaler = fitted_pipeline.named_steps['scaler']
        # fitted_log_transformer = fitted_pipeline.named_steps['log']
        # num_continuous = len(self.continuous_column)
        # processed_cont_data = processed_data[:, :num_continuous]
        # original_cat_data = processed_data[:, num_continuous:]
        # unscaled_data = fitted_scaler.inverse_transform(processed_cont_data)
        # restored_cont_data = fitted_log_transformer.inverse_transform(unscaled_data)
        # restored_data_array = np.concatenate([restored_cont_data, original_cat_data], axis=1)
        # current_columns = self.continuous_column + self.category_column
        # temp_df = pd.DataFrame(restored_data_array, columns=current_columns)
        # restored_df = temp_df[self.data_drop_fillna_raw.columns]

        self.mixed_transform=mixed_transform

    def save_scaler_params(self,scaler_save_path):
        os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
        self.scaler_save_path=scaler_save_path
        joblib.dump(self.scaler, self.scaler_save_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not self.mixed_transform:
            row = self.data_drop_fillna_log.iloc[idx]
            true_table=self.true_table.iloc[idx].astype('float32').values
            #true_table=(~self.data_drop.iloc[idx].isna()).astype('float32').values
            X = row.values.astype('float32')
            # 归一化（只特征归一化，不归一化y）
            X = self.scaler.transform([X])[0]
            X=X.astype('float32')
            return torch.from_numpy(X), torch.from_numpy(true_table)
        else:
            row =self.data_drop_fillna_log1cont_norm.iloc[idx]
            true_table = self.true_table.iloc[idx].astype('float32').values
            #true_table = (~self.data_drop.iloc[idx].isna()).astype('float32').values # cont first, cat second
            X=row[self.data_drop_fillna.columns]
            X = X.astype('float32')
            X_cont=X[self.continuous_column]
            X_disc=X[self.category_column]


            return torch.from_numpy(X.values), torch.from_numpy(true_table)



# test code
# mydataset=MyDataset('datasheets/combine_large_xls/combined_large_excel_v3.1.csv')
# result=mydataset[2]
# original_value=mydataset.scaler.inverse_transform([result])
# original_value_test=mydataset.data_drop.iloc[2].values.astype('float32')
# print('test end')