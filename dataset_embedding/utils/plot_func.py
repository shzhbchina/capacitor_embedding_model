import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
# def plot_zdist(plt_data,dim=0):
#     #latent_dim = test_output_dataset_z.shape[1]
#     #plt_data=test_output_dataset_z.detach().numpy()
#     i = dim  # max range(latent_dim)
#     plt.figure()
#     plt.hist(plt_data[:, i], bins=50, density=True, alpha=0.6, label='encoder z')
#     # 标准正态分布曲线
#     x = np.linspace(-2, 2, 100)
#     plt.plot(x, norm.pdf(x, 0, 1), 'r--', label='N(0,1)')
#     plt.title(f'Latent dim {i}')
#     plt.legend()
#     plt.show()
#     return 0


def plot_zdist(plt_data, dim=0, bins=50):
    """
    绘制潜在空间分布的直方图（1维）或二维直方图（2维）。

    参数:
        plt_data: ndarray, shape (n_samples, n_latent_dim)
        dim: int 或 tuple/list[int]，要绘制的维度
        bins: 直方图分箱数
    """
    # 如果 dim 是单个 int → 一维直方图
    if isinstance(dim, int):
        i = dim
        plt.figure()
        plt.hist(plt_data[:, i], bins=bins, density=True, alpha=0.6, label='encoder z')

        # 标准正态分布曲线
        x = np.linspace(-2, 2, 100)
        plt.plot(x, norm.pdf(x, 0, 1), 'r--', label='N(0,1)')
        plt.title(f'Latent dim {i}')
        plt.xlabel(f'z[{i}]')
        plt.ylabel('Density')
        plt.legend()
        plt.show()

    # 如果 dim 是两个维度 → 二维直方图
    elif isinstance(dim, (tuple, list)) and len(dim) == 2:
        i, j = dim
        plt.figure()
        plt.hist2d(plt_data[:, i], plt_data[:, j], bins=bins, cmap='Blues', density=True)
        plt.colorbar(label='Density')
        plt.xlabel(f'z[{i}]')
        plt.ylabel(f'z[{j}]')
        plt.title(f'Latent dims {i} vs {j}')
        plt.show()

    else:
        raise ValueError("dim 必须是单个 int（1维直方图）或包含两个 int 的 tuple/list（二维直方图）")

    return 0