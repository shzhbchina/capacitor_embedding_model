import torch
from torch.utils.data import DataLoader,random_split
import os
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
from dataset_embedding.utils.random_mask import random_mask


class PreTrainer():
    def __init__(self,model,loss_function,train_dataset,valid_dataset,training_config,define_diffusion=False):
        self.model= model
        self.train_dataset=train_dataset
        self.valid_dataset=valid_dataset
        self.loss_function=loss_function
        self.training_config=training_config
        self.define_diffusion=define_diffusion

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



        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # --- 關鍵修正：在加載完 scheduler 狀態後，立刻將其推進到正確的 epoch ---
            # 這樣可以確保優化器中的學習率在訓練開始前就是正確的
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                # 注意：PyTorch 1.1.0 之後的版本，load_state_dict 會自動恢復 epoch，
                # 但為了確保兼容性和邏輯清晰，我們保持這個邏輯。
                # scheduler.step(checkpoint['epoch']) # 對於 ReduceLROnPlateau

            start_epoch = checkpoint['epoch'] + 1
            best_valid_loss = checkpoint.get('best_valid_loss', float('inf'))
            print(f"成功加載檢查點，將從 Epoch {start_epoch} 開始訓練。")
            # 如果您想確認，可以在這裡打印一下學習率
            print(f"恢復後的學習率為: {optimizer.param_groups[0]['lr']:.6f}")



        for epoch in range(start_epoch, num_epoch):
            loss_tot,recon_loss_tot,mmd_loss_tot=0,0,0
            dataset_length=len(train_loader.dataset)
            model.train()
            for data,true_table in train_loader:
                if self.training_config['gauss_noise'] is not None:
                    data=data+torch.randn_like(data)*self.training_config['gauss_noise']
                data=data.to(device)
                true_table=true_table.to(device)
                data_mask,masked_table=random_mask(data,mask_ratio=0.3,mask_chance=0.9)
                effective_mask=masked_table.bool()+~true_table.bool()#invalid or masked data
                cat_masked_data=torch.concat([data_mask,~effective_mask],dim=1)#true_table
                if not self.define_diffusion:
                    [recon_batch, pred_z] = model(cat_masked_data)
                    [loss,recon_loss,mmd_loss] = self.loss_function(recon_batch,data,true_table=~effective_mask.bool())
                else:
                    [recon_batch, pred_z,  noise,pred_noise,x_noisy]=model(cat_masked_data, training_diffusion=True)
                    [loss, recon_loss, mmd_loss] = self.loss_function(recon_batch, data, pred_noise, noise,
                                                                      ~effective_mask.bool())
                optimizer.zero_grad()
                loss.backward()
                #batch_size=data.size(0)
                #dim_size = data.size(1)
                loss_tot += loss.item()/dataset_length*train_batch_size
                recon_loss_tot += recon_loss.item()/dataset_length*train_batch_size
                mmd_loss_tot += mmd_loss.item()/dataset_length*train_batch_size
                optimizer.step()
            print(f'Epoch {epoch}, Train Loss: {loss_tot:.6f}',end='')
            writer.add_scalars('Train_Loss/train', {'total_loss':loss_tot,
                                                   'recon_loss':recon_loss_tot,'mmd_loss':mmd_loss_tot}, epoch)


            valid_loss_tot,valid_recon_loss_tot,valid_mmd_loss_tot,valid_mean=0,0,0,0
            valid_mean_vc=0
            dataset_length = len(valid_loader.dataset)
            model.eval()
            for data,true_table in valid_loader:  # data应为归一化后的tensor
                with (torch.no_grad()):
                    data = data.to(device)
                    true_table = true_table.to(device)
                    print(data.device)
                    data_mask, masked_table = random_mask(data, mask_ratio=0.1, mask_chance=0.9)
                    effective_mask=masked_table.bool()+~true_table.bool()  # valid and masked data
                    cat_masked_data = torch.concat([data_mask, ~effective_mask], dim=1) #true_table
                    if not self.define_diffusion:
                        [recon_batch, pred_z] = model(cat_masked_data)
                        [loss, recon_loss, mmd_loss] = self.loss_function(recon_batch, data,true_table=~effective_mask.bool())
                    else:
                        [recon_batch, pred_z, noise, pred_noise, x_noisy] = model(cat_masked_data,
                                                                                  training_diffusion=True)
                        [loss, recon_loss, mmd_loss] = self.loss_function(recon_batch, data, pred_noise, noise,
                                                                          ~effective_mask.bool())
                    # batch_size = data.size(0)
                    # dim_size=data.size(1)
                    valid_loss_tot += loss.item() / dataset_length*data.shape[0]
                    valid_recon_loss_tot += recon_loss.item() / dataset_length*data.shape[0]
                    valid_mmd_loss_tot += mmd_loss.item() / dataset_length*data.shape[0]
                    valid_mean+=torch.mean(((recon_batch[0]-data)*true_table)**2)/dataset_length*data.shape[0]
                    valid_mean_vc += torch.mean((((recon_batch[0] - data) * true_table) ** 2)[:,0:2])/ dataset_length * data.shape[
                        0]
            print(f' Valid Loss: {valid_loss_tot:.6f}, mean err {valid_mean:.6f},{valid_mean_vc:.6f}')
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

            # 如果當前的驗證損失比之前的最佳損失還要低，就儲存為最佳模型
            if valid_loss_tot < best_valid_loss:
                best_valid_loss = valid_loss_tot
                torch.save(model.state_dict(), best_model_path)
                #print(f'已儲存新的最佳模型，驗證損失為: {best_valid_loss:.6f}')
        writer.close()

        #  plot
        latent_dim = pred_z.shape[1]
        i=3 #max range(latent_dim)
        plt.figure()
        plt.hist(pred_z[:, i], bins=50, density=True, alpha=0.6, label='encoder z')
        # 标准正态分布曲线
        x = np.linspace(-2, 2, 100)
        plt.plot(x, norm.pdf(x, 0, 1), 'r--', label='N(0,1)')
        plt.title(f'Latent dim {i}')
        plt.legend()
        plt.show()