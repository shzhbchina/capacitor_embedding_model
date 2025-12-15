import torch
from torch.utils.data import DataLoader,random_split
import os
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np


class BaseTrainer():
    def __init__(self,model,loss_function,train_dataset,valid_dataset,training_config,define_diffusion=False,input_concate_true_table=False):
        self.model= model
        self.train_dataset=train_dataset
        self.valid_dataset=valid_dataset
        self.loss_function=loss_function
        self.training_config=training_config
        self.define_diffusion=define_diffusion
        self.input_concate_true_table=input_concate_true_table

    def train(self):
        num_epoch=self.training_config['num_epoch']
        device=self.training_config['device']
        model=self.model.to(device)
        optimizer_name=self.training_config['optimizer_name']
        optimizer_lr = self.training_config['optimizer_lr']
        optimizer_weight_decay=self.training_config['optimizer_decay_weight']
        scheduler_name=self.training_config['scheduler_name']
        save_name=self.training_config['save_name']
        if self.training_config['batch_size'] is not None:
            train_batch_size=self.training_config['batch_size']
        else:
            train_batch_size=128

        optimizer_classes = {
            'Adam': torch.optim.Adam,
            'SGD': torch.optim.SGD,
            'AdamW': torch.optim.AdamW,
            'RMSprop': torch.optim.RMSprop
        }
        if optimizer_name not in optimizer_classes:
            raise ValueError(f"不支援的優化器名稱: {optimizer_name}")
        optimizer_cls = optimizer_classes[optimizer_name]
        optimizer_params = {'lr': optimizer_lr}
        if optimizer_name == 'AdamW':
            optimizer_params['weight_decay'] = optimizer_weight_decay
        optimizer = optimizer_cls(model.parameters(), **optimizer_params) #**是解包符，将字典解包为参数


        scheduler_classes={'CosineAnnealingLR':torch.optim.lr_scheduler.CosineAnnealingLR,
                    'StepLR':torch.optim.lr_scheduler.StepLR,
                   'ReduceLROnPlateau':torch.optim.lr_scheduler.ReduceLROnPlateau,
                   'OneCycleLR':torch.optim.lr_scheduler.OneCycleLR}
        if scheduler_name not in scheduler_classes:
            raise ValueError(f"不支援的學習率調整器名稱: {scheduler_name}")
        scheduler_cls = scheduler_classes[scheduler_name]
        scheduler_params = self.training_config['scheduler_params'][scheduler_name]

        scheduler = scheduler_cls(optimizer, **scheduler_params)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch)
        #scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)#classical *gamma every setp_size
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1) #if not improve after patience
        #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, total_steps=100) #warmup

        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_script_dir=os.path.dirname(current_script_dir)



        save_dir = os.path.join(parent_script_dir, 'save',save_name)
        os.makedirs(save_dir + '', exist_ok=True)
        checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
        best_model_path = os.path.join(save_dir, 'best_model.pth')
        writer = SummaryWriter(os.path.join(save_dir,'experiment'))

        #train_batch_size=128
        valid_batch_size=100000


        is_tensor_dataset_on_gpu = (device == 'cuda' and
                                    self.train_dataset.dataset.tensors[0].is_cuda)
        is_data_on_gpu = False
        # 情況一：傳入的是 random_split 後的 Subset 物件
        if device == 'cuda' and hasattr(self.train_dataset, 'dataset') and isinstance(self.train_dataset.dataset,
                                                                                      torch.utils.data.TensorDataset):
            if self.train_dataset.dataset.tensors[0].is_cuda:
                is_data_on_gpu = True
        # 情況二：傳入的是 TensorDataset 物件本身
        elif device == 'cuda' and isinstance(self.train_dataset, torch.utils.data.TensorDataset):
            if self.train_dataset.tensors[0].is_cuda:
                is_data_on_gpu = True

        if is_data_on_gpu:
            # 如果數據已在 GPU 上，num_workers 必須為 0
            print("偵測到 GPU 上的 TensorDataset，設定 num_workers=0")
            loader_args = {'num_workers': 0, 'pin_memory': False}  # pin_memory 在此情況下也無需開啟
        elif device == 'cuda':
            # 如果是普通的 Dataset (從硬碟讀取)，才使用多進程
            loader_args = {'num_workers': 12, 'pin_memory': True}
        else:
            # CPU 模式
            loader_args = {'num_workers': 0, 'pin_memory': False}

        train_loader = DataLoader(self.train_dataset, batch_size=train_batch_size, shuffle=True,**loader_args)
        valid_loader = DataLoader(self.valid_dataset, batch_size=valid_batch_size, shuffle=False,**loader_args)

        best_valid_loss = float('inf')  # 初始化一個無限大的最佳損失
        start_epoch=0
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_valid_loss = checkpoint.get('best_valid_loss', float('inf'))  # .get 提供向下相容性
            print(f"成功加載檢查點，將從 Epoch {start_epoch} 開始訓練。")



        # if os.path.exists(checkpoint_path):
        #     checkpoint = torch.load(checkpoint_path, map_location=device)
        #     model.load_state_dict(checkpoint['model_state_dict'])
        #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #
        #     # --- 關鍵修正：在加載完 scheduler 狀態後，立刻將其推進到正確的 epoch ---
        #     # 這樣可以確保優化器中的學習率在訓練開始前就是正確的
        #     if 'scheduler_state_dict' in checkpoint:
        #         scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        #         # 注意：PyTorch 1.1.0 之後的版本，load_state_dict 會自動恢復 epoch，
        #         # 但為了確保兼容性和邏輯清晰，我們保持這個邏輯。
        #         # scheduler.step(checkpoint['epoch']) # 對於 ReduceLROnPlateau
        #
        #     start_epoch = checkpoint['epoch'] + 1
        #     best_valid_loss = checkpoint.get('best_valid_loss', float('inf'))
        #     print(f"成功加載檢查點，將從 Epoch {start_epoch} 開始訓練。")
        #     # 如果您想確認，可以在這裡打印一下學習率
        #     print(f"恢復後的學習率為: {optimizer.param_groups[0]['lr']:.6f}")



        for epoch in range(start_epoch, num_epoch):
            loss_tot,recon_loss_tot,mmd_loss_tot=0,0,0
            train_mean, train_mean_vc,train_mean_esr,train_max_esr = 0, 0,0,0
            dataset_length=len(train_loader.dataset)
            model.train()
            for data,true_table in train_loader:
                if self.training_config['gauss_noise'] is not None:
                    data=data+torch.randn_like(data)*self.training_config['gauss_noise']
                if self.training_config['uniform_noise'] is not None:
                    data = data + (torch.rand_like(data) * 2 - 1) *self.training_config['uniform_noise']
                data=data.to(device)
                true_table=true_table.to(device)
                if not self.define_diffusion:
                    if self.input_concate_true_table:
                        [recon_batch, pred_z] = model(torch.cat([data,true_table],dim=1))
                    else:
                        [recon_batch, pred_z] = model(data)
                    [loss,recon_loss,mmd_loss] = self.loss_function(recon_batch,data,pred_z,true_table.clone())
                else:
                    [recon_batch, pred_z,  noise,pred_noise,x_noisy]=model(data, training_diffusion=True)
                    [loss, recon_loss, mmd_loss] = self.loss_function(recon_batch, data, pred_z, pred_noise, noise,
                                                                      true_table.clone())
                optimizer.zero_grad()
                loss.backward()
                #batch_size=data.size(0)
                #dim_size = data.size(1)
                loss_tot += loss.item()/dataset_length*train_batch_size
                recon_loss_tot += recon_loss.item()/dataset_length*train_batch_size
                mmd_loss_tot += mmd_loss.item()/dataset_length*train_batch_size
                train_mean += torch.mean(((recon_batch[0] - data) * true_table) ** 2) / dataset_length * data.shape[0]
                train_mean_vc += torch.mean((((recon_batch[0] - data) * true_table) ** 2)[:, 0:2]) / dataset_length * \
                                 data.shape[0]
                train_mean_esr += torch.mean((((recon_batch[0] - data) * true_table) ** 2)[:, 13]) / dataset_length * \
                                 data.shape[0]
                train_max_esr_batch=torch.max((((recon_batch[0] - data) * true_table) ** 2)[:, 1])
                train_max_esr=torch.max(torch.tensor((train_max_esr_batch,train_max_esr)))

                optimizer.step()
            print(f'Epoch {epoch}, Train Loss: {loss_tot:.6f}',end='')
            print(f' Train Loss: {loss_tot:.6f}, mean err {train_mean:.6f}, vc {train_mean_vc:.6f},esr {train_mean_esr:.6f},est_m{train_max_esr:.6f}',end='')
            writer.add_scalars('Train_Loss/train', {'total_loss':loss_tot,
                                                   'recon_loss':recon_loss_tot,'mmd_loss':mmd_loss_tot}, epoch)


            valid_loss_tot,valid_recon_loss_tot,valid_mmd_loss_tot=0,0,0
            valid_mean,valid_mean_vc,valid_mean_esr,valid_max_esr=0,0,0,0
            dataset_length = len(valid_loader.dataset)
            model.eval()
            for data,true_table in valid_loader:  # data应为归一化后的tensor
                with (torch.no_grad()):
                    data = data.to(device)
                    true_table = true_table.to(device)
                    print(data.device)
                    if not self.define_diffusion:
                        if self.input_concate_true_table:
                            [recon_batch, pred_z] = model(torch.cat([data, true_table], dim=1))
                        else:
                            [recon_batch, pred_z] = model(data)
                        [loss, recon_loss, mmd_loss] = self.loss_function(recon_batch, data, pred_z,true_table.clone())
                    else:
                        [recon_batch, pred_z,  noise,pred_noise,x_noisy]=model(data, training_diffusion=True)
                        [loss, recon_loss, mmd_loss] = self.loss_function(recon_batch, data, pred_z, pred_noise, noise,
                                                                          true_table.clone())
                    # batch_size = data.size(0)
                    # dim_size=data.size(1)
                    valid_loss_tot += loss.item() / dataset_length*data.shape[0]
                    valid_recon_loss_tot += recon_loss.item() / dataset_length*data.shape[0]
                    valid_mmd_loss_tot += mmd_loss.item() / dataset_length*data.shape[0]
                    valid_mean+=torch.mean(((recon_batch[0]-data)*true_table)**2)/dataset_length*data.shape[0]
                    valid_mean_vc += torch.mean((((recon_batch[0] - data) * true_table) ** 2)[:,0:2])/ dataset_length * data.shape[
                        0]
                    valid_mean_esr += torch.mean(
                        (((recon_batch[0] - data) * true_table) ** 2)[:, 13]) / dataset_length * \
                                      data.shape[0]
                    valid_max_esr += torch.max(
                        (((recon_batch[0] - data) * true_table) ** 2)[:, 1]) / dataset_length * \
                                      data.shape[0]
            print(f' Valid Loss: {valid_loss_tot:.6f}, mean err {valid_mean:.6f}, vc{valid_mean_vc:.6f},esr{valid_mean_esr:.6f},esr_m{valid_max_esr:.6f}')
            writer.add_scalars('Train_Loss/train', {'total_loss':valid_loss_tot,
                                                   'recon_loss':valid_recon_loss_tot,'mmd_loss':valid_mmd_loss_tot}, epoch)
            if scheduler_name=='ReduceLROnPlateau':
                scheduler.step(loss_tot)
            else :
                scheduler.step()



            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_valid_loss': best_valid_loss
            }, checkpoint_path)
            # 0:voltage,1:cap,3-5:shape:rec,6-7:shape:cylindar,8:temp,9:life,10:type,12:price,13:ESR,38:ripple
            # check_column=[0,1,3,4,5,6,7,8,9,10,12,13,38]
            # 如果當前的驗證損失比之前的最佳損失還要低，就儲存為最佳模型
            save_new=True
            if (valid_loss_tot < best_valid_loss)or save_new:
                best_valid_loss = valid_loss_tot
                torch.save(model.state_dict(), best_model_path)
                #print(f'已儲存新的最佳模型，驗證損失為: {best_valid_loss:.6f}')
        writer.close()

        #  1. plot norm
        # latent_dim = pred_z.shape[1]
        # i=3 #max range(latent_dim)
        # plt.figure()
        # plt.hist(pred_z[:, i], bins=50, density=True, alpha=0.6, label='encoder z')
        # # 标准正态分布曲线
        # x = np.linspace(-2, 2, 100)
        # plt.plot(x, norm.pdf(x, 0, 1), 'r--', label='N(0,1)')
        # plt.title(f'Latent dim {i}')
        # plt.legend()
        # plt.show()

        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']  # 建议使用 Arial 或 Times New Roman
        plt.rcParams['font.weight'] = 'bold'  # 全局粗体
        plt.rcParams['axes.labelweight'] = 'bold'  # 轴标签粗体
        latent_dim = pred_z.shape[1]
        i = 7
        fig, ax = plt.subplots(figsize=(8, 6))
        n, bins, patches = ax.hist(pred_z[:, i], bins=60, density=True,
                                   color='#5D9CEC', alpha=0.7,
                                   edgecolor='white', linewidth=0.5,
                                   label='Encoder $z$')
        x_min, x_max = ax.get_xlim()
        x = np.linspace(x_min, x_max, 200)  # 生成更平滑的点
        ax.plot(x, norm.pdf(x, 0, 1), color='#FF4500', linestyle='--',
                linewidth=3.5, label='Prior N(0,1)')
        ax.set_title(f'Latent Dimension {i} Distribution', fontsize=22, fontweight='bold', pad=20)
        ax.set_xlabel("Latent Value", fontsize=18, labelpad=10)
        ax.set_ylabel("Density", fontsize=18, labelpad=10)
        ax.tick_params(axis='both', which='major', labelsize=16, width=2, length=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.grid(True, linestyle='--', alpha=0.4, color='gray', zorder=0)
        plt.legend(fontsize=16, frameon=False, loc='upper right')
        plt.tight_layout()
        plt.show()














        #2. plot box plot
        stats_src = {
            "Voltage": (0.014, 0.030, 0.049, 0.222),
            "Cap": (0.011, 0.024, 0.042, 0.177),
            "temp": (0.012, 0.026, 0.047, 0.508),
            "Life": (0.018, 0.037, 0.062, 0.357),
            "Price": (0.000, 0.015, 0.064, 0.361),
            "ESR": (0.006, 0.015, 0.027, 0.144),
            "Ripple": (0.011, 0.024, 0.042, 0.256),
        }
        stats = []
        for label, (q1, med, q3, max_v) in stats_src.items():
            stats.append({
                "label": label,
                "whislo": 0.0,  # 按要求设定最小值为 0
                "q1": q1,
                "med": med,
                "q3": q3,
                "whishi": max_v,
                "fliers": []  # 没有离群点
            })
        # fig, ax = plt.subplots(figsize=(8, 6))
        # ax.bxp(stats, showfliers=False)  # 使用统计量直接画箱线图
        # ax.set_ylabel("Value")
        # ax.set_title("Boxplot from given quartiles (min fixed at 0)")
        # ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        # plt.tight_layout()
        # ax.set_title("Boxplot of Attributes", fontsize=20)
        # ax.set_xlabel("Attributes", fontsize=16)
        # ax.set_ylabel("Values", fontsize=16)
        # ax.tick_params(axis='both', which='major', labelsize=14)
        # plt.show()

        # 设置全局字体为粗体，且使用无衬线字体 (如 Arial/Helvetica)
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['axes.labelweight'] = 'bold'
        fig, ax = plt.subplots(figsize=(12, 7))
        boxplots = ax.bxp(stats,
                          patch_artist=True,
                          showfliers=False,
                          vert=True,
                          widths=0.65)  # 稍微再宽一点点
        box_face_color = '#AEC7E8'  # 浅蓝填充
        box_edge_color = '#003366'  # 深蓝近黑 (加深颜色以增加对比)
        median_color = '#FF4500'  # 鲜艳的橙红色 (OrangeRed)
        lw_box = 2.5
        lw_whisker = 2.0
        lw_median = 3.0
        for box in boxplots['boxes']:
            box.set(facecolor=box_face_color, edgecolor=box_edge_color, linewidth=lw_box)
        for whisker in boxplots['whiskers']:
            whisker.set(color='#333333', linewidth=lw_whisker, linestyle='-')
        for cap in boxplots['caps']:
            cap.set(color='#333333', linewidth=lw_whisker)
        for median in boxplots['medians']:
            median.set(color=median_color, linewidth=lw_median)  # 中位数线最粗
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        spine_width = 2.5
        ax.spines['left'].set_linewidth(spine_width)
        ax.spines['bottom'].set_linewidth(spine_width)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.tick_params(axis='both', which='major',
                       width=2.5,  # 刻度线变粗
                       length=8,  # 刻度线变长
                       labelsize=16,  # 字号
                       color='black',
                       labelcolor='black')
        ax.set_title("Distribution of Attributes", fontsize=24, fontweight='bold', pad=20)
        ax.set_xlabel("Attributes", fontsize=18, fontweight='bold', labelpad=10)
        ax.set_ylabel("Values", fontsize=18, fontweight='bold', labelpad=10)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='gray', zorder=0, linewidth=1.5)
        plt.tight_layout()
        plt.show()






        #3.calculate dcor of pred_z
        pred_z_numpy=pred_z.detach().numpy()
        N, D = pred_z_numpy.shape
        dcor_matrix = np.zeros((D, D))
        print("正在计算 dCor 矩阵...")
        # 双重循环遍历每一对维度
        for i in range(D):
            for j in range(D):
                # 对角线直接设为 1
                if i == j:
                    dcor_matrix[i, j] = 1.0
                    continue
                # 获取两个维度的数据向量 (长度为 N)
                X = pred_z_numpy[:, i]
                Y = pred_z_numpy[:, j]
                # --- 步骤 A: 计算欧氏距离矩阵 (N x N) ---
                # 利用广播机制: (N, 1) - (1, N) -> (N, N)
                # 对于一维标量数据，欧氏距离就是绝对值差
                A = np.abs(X[:, None] - X[None, :])
                B = np.abs(Y[:, None] - Y[None, :])
                # --- 步骤 B: 距离矩阵中心化 (Double Centering) ---
                # 公式: A_centered = A - 行均值 - 列均值 + 总均值
                A_mean_row = A.mean(axis=1, keepdims=True)
                A_mean_col = A.mean(axis=0, keepdims=True)
                A_mean_all = A.mean()
                A_centered = A - A_mean_row - A_mean_col + A_mean_all
                B_mean_row = B.mean(axis=1, keepdims=True)
                B_mean_col = B.mean(axis=0, keepdims=True)
                B_mean_all = B.mean()
                B_centered = B - B_mean_row - B_mean_col + B_mean_all
                # --- 步骤 C: 计算距离协方差和方差 ---
                # dCov(X,Y) 的平方
                dcov2_xy = np.mean(A_centered * B_centered)
                # dVar(X) 的平方 和 dVar(Y) 的平方
                dvar2_xx = np.mean(A_centered * A_centered)
                dvar2_yy = np.mean(B_centered * B_centered)
                # --- 步骤 D: 计算 dCor ---
                # dCor = sqrt(dCov^2) / sqrt(sqrt(dVar^2_x) * sqrt(dVar^2_y))
                # 加一个极小值 1e-10 防止除以零
                dcor_val = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dvar2_xx) * np.sqrt(dvar2_yy) + 1e-10)
                dcor_matrix[i, j] = dcor_val
        # ==========================================
        # 3. 输出与可视化
        # ==========================================
        print("\nDistance Correlation Matrix:")
        print(np.round(dcor_matrix, 4))
        import seaborn as sns
        if 'dcor_matrix' not in locals():
            dcor_matrix = np.random.rand(5, 5)
            np.fill_diagonal(dcor_matrix, 1.0)  # 对角线肯定是1
        plt.figure(figsize=(8, 6))  # 设置画布大小
        sns.heatmap(dcor_matrix, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1)
        plt.title("Distance Correlation Matrix (dCor)", fontsize=14)
        plt.xlabel("Dimension Index")
        plt.ylabel("Dimension Index")
        plt.show()
