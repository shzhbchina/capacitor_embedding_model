import torch
from torch.utils.data import DataLoader,random_split

import os
from torch.utils.tensorboard import SummaryWriter

class BaseTrainer():
    def __init__(self,model,loss_function,train_dataset,valid_dataset,training_config):
        self.model= model
        self.train_dataset=train_dataset
        self.valid_dataset=valid_dataset
        self.loss_function=loss_function
        self.training_config=training_config

    def train(self):
        num_epoch=self.training_config['num_epoch']
        device=self.training_config['device']
        model=self.model
        optimizer_name=self.training_config['optimizer_name']
        optimizer_lr = self.training_config['optimizer_lr']
        optimizer_weight_decay=self.training_config['optimizer_decay_weight']
        scheduler_name=self.training_config['scheduler_name']
        save_name=self.training_config['save_name']

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

        train_loader = DataLoader(self.train_dataset, batch_size=128, shuffle=True)
        valid_loader = DataLoader(self.valid_dataset, batch_size=128, shuffle=False)

        best_valid_loss = float('inf')  # 初始化一個無限大的最佳損失
        start_epoch=0
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_valid_loss = checkpoint.get('best_valid_loss', float('inf'))  # .get 提供向下相容性
            print(f"成功加載檢查點，將從 Epoch {start_epoch} 開始訓練。")

        for epoch in range(start_epoch, num_epoch):
            loss_tot,recon_loss_tot,mmd_loss_tot=0,0,0
            dataset_length=len(train_loader.dataset)
            for data,true_table in train_loader:
                data=data.to(device)
                true_table=true_table.to(device)
                [recon_batch, pred_z] = model(data)
                [loss,recon_loss,mmd_loss] = self.loss_function(recon_batch,data,pred_z,true_table)
                optimizer.zero_grad()
                loss.backward()
                loss_tot += loss.item()/dataset_length
                recon_loss_tot += recon_loss.item()/dataset_length
                mmd_loss_tot += mmd_loss.item()/dataset_length
                optimizer.step()
            print(f'Epoch {epoch}, Train Loss: {loss_tot:.6f}',end='')
            writer.add_scalars('Train_Loss/train', {'total_loss':loss_tot,
                                                   'recon_loss':recon_loss_tot,'mmd_loss':mmd_loss_tot}, epoch)


            loss_tot,recon_loss_tot,mmd_loss_tot=0,0,0
            dataset_length = len(valid_loader.dataset)
            model.eval()
            for data,true_table in valid_loader:  # data应为归一化后的tensor
                with torch.no_grad():
                    data = data.to(device)
                    true_table = true_table.to(device)
                    [recon_batch, pred_z] = model(data)
                    [loss, recon_loss, mmd_loss] = self.loss_function(recon_batch, data, pred_z,true_table)
                    loss_tot += loss.item() / dataset_length
                    recon_loss_tot += recon_loss.item() / dataset_length
                    mmd_loss_tot += mmd_loss.item() / dataset_length
            print(f' Valid Loss: {loss_tot:.6f}')
            writer.add_scalars('Train_Loss/train', {'total_loss':loss_tot,
                                                   'recon_loss':recon_loss_tot,'mmd_loss':mmd_loss_tot}, epoch)
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
            if loss_tot < best_valid_loss:
                best_valid_loss = loss_tot
                torch.save(model.state_dict(), best_model_path)
                #print(f'已儲存新的最佳模型，驗證損失為: {best_valid_loss:.6f}')
        writer.close()