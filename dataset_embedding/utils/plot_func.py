import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
def plot_zdist(plt_data,dim=0):
    #latent_dim = test_output_dataset_z.shape[1]
    #plt_data=test_output_dataset_z.detach().numpy()
    i = dim  # max range(latent_dim)
    plt.figure()
    plt.hist(plt_data[:, i], bins=50, density=True, alpha=0.6, label='encoder z')
    # 标准正态分布曲线
    x = np.linspace(-2, 2, 100)
    plt.plot(x, norm.pdf(x, 0, 1), 'r--', label='N(0,1)')
    plt.title(f'Latent dim {i}')
    plt.legend()
    plt.show()
    return 0