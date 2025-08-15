
import numpy as np
from sklearn.mixture import GaussianMixture


def cal_GMM(z_train):
    # 1) 标准化 z（建议与KDE一致）
    z_mu, z_std = z_train.mean(axis=0), z_train.std(axis=0) + 1e-8
    z_train_std = (z_train - z_mu) / z_std

    # 2) 选 K（高斯个数）和协方差类型
    #   - 先用 BIC/AIC 选择 K（例如 2~16）
    candidates = []
    bics = []
    for K in [2, 4, 6, 8, 12, 16]:
        gmm = GaussianMixture(n_components=K, covariance_type='full', reg_covar=1e-4, max_iter=int(500))
        gmm.fit(z_train_std)
        candidates.append(gmm)
        bics.append(gmm.bic(z_train_std))
    gmm = candidates[int(np.argmin(bics))]

    return gmm,z_mu,z_std

def density_penalty_gmm(z, z_mu, z_std, gmm,clip=50.0):
    z_std_ = (z - z_mu) / z_std
    logp = gmm.score_samples(z_std_)   # 对数密度
    return np.clip(-logp, 0.0, clip)



