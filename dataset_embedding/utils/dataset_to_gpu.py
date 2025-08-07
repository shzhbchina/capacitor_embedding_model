import torch
from torch.utils.data import TensorDataset

def dataset_to_gpu(processed_data_np,true_table_df):
    # 1. 將處理好的 NumPy array 轉換為一個大的 Torch Tensor
    full_data_tensor = torch.tensor(processed_data_np.values, dtype=torch.float32)
    # true_table 也一樣處理
    true_table_tensor = torch.tensor(true_table_df.values, dtype=torch.float32)

    # 2. (關鍵) 一次性將整個 Tensor 移動到 GPU
    print("正在將數據移動到 GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_data_gpu = full_data_tensor.to(device)
    true_table_gpu = true_table_tensor.to(device)
    print(f'數據已在 {device} 上。')

    # 3. 使用 TensorDataset 將 GPU 上的 Tensor 包裝起來
    # TensorDataset 會直接從 GPU Tensor 中切片，沒有額外的 CPU->GPU 傳輸
    gpu_dataset = TensorDataset(full_data_gpu, true_table_gpu)
    return gpu_dataset