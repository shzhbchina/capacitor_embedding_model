import numpy as np
import torch
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

latent_dim = 16
def curve_selected_dimension(test_query_datasheet,base_z,z_dim,show_column_name,samples=100):
    seed_gen = torch.Generator().manual_seed(42)
    # 固定其余z，测试第0维的变化
    z_fixed = base_z
    z_range = np.linspace(-3, 3, samples)
    outputs = []

    for zi in z_range:
        z_cur = z_fixed.clone()
        z_cur[0, z_dim] = zi  # 修改第0维
        out_df = test_query_datasheet.find_component_from_latent_space(z_cur, mixed_transform=True)
        # 取某一列特征作对比，比如“Voltage”
        outputs.append(out_df[show_column_name].values[0])  # 按需换列名

    plt.figure(figsize=(7,4))
    plt.plot(z_range, outputs, marker='o')
    plt.xlabel(f'z{z_dim}')
    plt.ylabel(show_column_name)
    plt.title(f'生成数据随z{z_dim}变化曲线（是否平滑）')
    plt.grid(True)
    plt.show()

#reference usage
# seed_gen = torch.Generator().manual_seed(42)
# z_fixed = torch.randn(1, latent_dim, generator=seed_gen)
# base_z=z_fixed
# z_dim=0
# show_column_name=['Rated Voltage /V']
# curve_selected_dimension(test_query_datasheet,base_z,z_dim,show_column_name,samples=100)