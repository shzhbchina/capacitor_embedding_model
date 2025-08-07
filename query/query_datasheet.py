#from dataset_embedding.data.dataset import MyDataset
import joblib,os
#from dataset_embedding.models.WassersteinAutoEncoder import WAE
import torch
import pandas as pd
import numpy as np


class query_datasheet():
    def __init__(self,model,dataset,scaler_save_path=None):
        self.model=model
        self.dataset= dataset
        if scaler_save_path is not None:
            self.scaler=joblib.load(scaler_save_path)
        print('end')

    def find_component_from_latent_space(self,latent_vector,real_component_config=None,mixed_transform=False):
        self.model.eval()
        if latent_vector.dim()==1:
            latent_vector=latent_vector.unsqueeze(0)
        decode_vector=self.model.decode(latent_vector)
        if mixed_transform:
            # restored_data_array = self.dataset.preprocessor.inverse_transform(decode_vector) #cont first, cat second
            # restored_df = pd.DataFrame(restored_data_array, columns=self.dataset.continuous_column + self.dataset.category_column)
            # df = restored_df[self.data.data_drop_fillna.columns]

            fitted_pipeline = self.dataset.preprocessor.named_transformers_['cont']
            fitted_scaler = fitted_pipeline.named_steps['scaler']
            fitted_log_transformer = fitted_pipeline.named_steps['log']
            num_continuous = len(self.dataset.continuous_column)
            processed_cont_data = decode_vector[1]
            original_cat_data = decode_vector[2]
            unscaled_data = fitted_scaler.inverse_transform(processed_cont_data)
            restored_cont_data = fitted_log_transformer.inverse_transform(unscaled_data)
            restored_data_array = np.concatenate([restored_cont_data, original_cat_data], axis=1)
            current_columns = self.dataset.continuous_column + self.dataset.category_column
            temp_df = pd.DataFrame(restored_data_array, columns=current_columns)
            df = temp_df[self.dataset.data_drop_fillna_raw.columns]




        else:

            component_params=np.exp(self.scaler.inverse_transform(decode_vector.detach().numpy()))
            col_mask=self.dataset.col_mask
            cols=col_mask[col_mask].index.tolist()
            df = pd.DataFrame(component_params, columns=cols)

        # 如果沒有提供篩選條件，則直接返回模型生成的理想元件參數
        if not real_component_config:
            return df

        # --- 2. 根據條件篩選數據庫 ---
        top_k = real_component_config.get('top_k', 5)  # 使用 .get 提供預設值
        voltage_cons = real_component_config.get('voltage_constraint', 0)
        capacitance_cons = real_component_config.get('capacitance_constraint', 0)
        param_weight=real_component_config.get('param_weight', None)

        # 建立一個只包含相關特徵欄位的數據庫副本
        search_dataset = self.dataset.data_drop_fillna.copy()
        mask = (search_dataset['Rated Voltage /V'] >= voltage_cons) & \
               (search_dataset['Rated Capacitance /uF'] >= capacitance_cons)
        dataset_filtered = search_dataset[mask]

        if dataset_filtered.empty:
            print("警告：沒有任何數據庫條目滿足篩選條件。")
            return pd.DataFrame()  # 返回一個空的 DataFrame

        if len(df) > 1:
            print("警告：此函式目前只為第一筆生成數據進行搜尋。")

        target_vector = df.iloc[0].values  # 我們的目標元件參數 (模型生成的)


        diff = (dataset_filtered.values - target_vector)/target_vector
        diff_weight=diff*param_weight
        # 2. 計算每個條目(每一行)的 L2 範數，也就是歐氏距離
        distances = np.linalg.norm(diff_weight, axis=1)

        # 3. 找到距離最小的 top_k 個條目的索引
        # np.argsort() 會返回排序後原始索引
        top_k_indices = np.argsort(distances)[:top_k]

        # --- 4. 提取並返回結果 ---
        # 使用找到的索引，從篩選後的數據庫中提取出最接近的 k 個條目
        selected_top_k = dataset_filtered.iloc[top_k_indices]

        # (可選) 將計算出的距離也加入到結果中，方便查看
        selected_top_k_dist = distances[top_k_indices]


        return selected_top_k





