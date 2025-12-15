import joblib
import numpy as np
import os
import torch
# from dataset_embedding.models.MT_MixedWassersteinAutoEncoder import MainMaskedMixedWAE
#from dataset_embedding.models.MT_CMixedWassersteinAutoEncoder import CMainMaskedMixedWAE
from dataset_embedding.models.MT_RCMixedWassersteinAutoEncoder import RCMainMaskedMixedWAE
from dataset_embedding.data.dataset import MyDataset
from query.query_datasheet import query_datasheet
from scipy.interpolate import griddata, RegularGridInterpolator
import re
import pyswarms as ps
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from functools import partial
import pandas as pd
from dataset_embedding.utils.cal_GMM import density_penalty_gmm

class optimizer():
    def __init__(self,model,query_instance,gmm_path=None):
        self.model=model
        self.query_instance=query_instance
        column_name = self.query_instance.dataset.data_remain_columns
        self.Hz_deg_list=self.extract_freq_temp(column_name)
        if gmm_path is not None:
            self.load_gmm(gmm_path,100.0)

        print('end')
    def load_gmm(self,gmm_path,clip=100.0):
        payload=joblib.load(gmm_path)
        self.payload=payload
        self.density_penalty_gmm=partial(density_penalty_gmm,gmm=payload['gmm'],z_mu=payload['z_mu'],z_std=payload['z_std'],clip=clip)
    def extract_freq_temp(self,column_name):
        hz_list = []
        deg_list = []

        for col in column_name:
            # 匹配 Ripple_at_1000_Hz_30_deg 这种
            m = re.search(r'_(\d+)_Hz_(\d+)_deg', col)
            if m:
                hz_list.append(int(m.group(1)))
                deg_list.append(int(m.group(2)))

        hz_sorted = sorted(list(set(hz_list)))
        deg_sorted = sorted(list(set(deg_list)))
        return [hz_sorted, deg_sorted]

    def PSO(self, design_target,var_init_config, cost_weight, optimization_config=None):
        #best_z_inverse_gs,best_info=self.grid_search(design_target, var_init_config, cost_weight, optimization_config)
        particle_num=var_init_config['particle_num']
        init_scale=var_init_config['scale']
        init_distribution=var_init_config['var_distribution']
        init_shift=var_init_config['shift']
        dim=var_init_config['latent_dim']

        min_bounds = []
        max_bounds = []
        for name, shift, scale in zip(init_distribution, init_shift, init_scale):
            lower_bound = shift - scale
            upper_bound = shift + scale
            min_bounds.append(lower_bound)
            max_bounds.append(upper_bound)
        bounds = (np.array(min_bounds), np.array(max_bounds))

        init_particles=[]
        i=0
        for name,shift,scale in zip(init_distribution,init_shift,init_scale):
            if name == 'norm':
                sample = np.random.randn(particle_num)
            elif name == 'folded_norm':
                sample = np.abs(np.random.randn(particle_num))
            elif name=='uniform':
                sample = np.random.uniform(size=particle_num,low=-1, high=1)
            else:
                raise ValueError(f"Unsupported distribution {name}")
            particle_dim_i=sample*scale+ shift
            init_particles.append(particle_dim_i)
            i+=1
        init_particles=np.stack(init_particles,axis=1)
        particles=init_particles

        if self.payload is not None:
            gmm=self.payload['gmm']
            z_std=self.payload['z_std']
            z_mu=self.payload['z_mu']
            z_samp_std, _ = gmm.sample(particle_num)  # 在标准化坐标采样
            z_samp = z_samp_std * z_std + z_mu  # 还原到原 z 坐标
            particles=np.concatenate((z_samp,init_particles[:, -2:]),axis=1)
            lb = np.asarray(bounds[0], dtype=float)
            ub = np.asarray(bounds[1], dtype=float)
            particles = np.clip(particles, lb, ub)  # 对所有维度逐元素钳位到 [lb, ub]
            particles[:, -2:] = np.ceil(particles[:, -2:])  # 先四舍五入
            particles[:, -2:] = np.clip(particles[:, -2:], lb[-2:], ub[-2:])
            #cost=self.cost_function(particles,cost_weight,design_target)

            uniform_weights = np.ones(gmm.n_components) / gmm.n_components
            component = np.random.choice(np.arange(gmm.n_components), p=uniform_weights)
            z_samp_std = np.random.multivariate_normal(gmm.means_[component], gmm.covariances_[component])
            z_samp_std, comp_ids = self.sample_uniform_from_gmm(gmm, n_samples=particle_num, random_state=42)
            z_samp = z_samp_std * z_std + z_mu  # 还原到原 z 坐标
            particles=np.concatenate((z_samp,init_particles[:, -2:]),axis=1)
            lb = np.asarray(bounds[0], dtype=float)
            ub = np.asarray(bounds[1], dtype=float)
            particles = np.clip(particles, lb, ub)  # 对所有维度逐元素钳位到 [lb, ub]
            particles[:, -2:] = np.ceil(particles[:, -2:])  # 先四舍五入
            particles[:, -2:] = np.clip(particles[:, -2:], lb[-2:], ub[-2:])


        cost_function_partial = partial(
            self.cost_function,
            cost_weight=cost_weight,
            design_target=design_target
        )


        #dim = 18  # 每维边界
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}  # PSO参数，常用即可
        options ={'c1': 1.0, 'c2': 1.0, 'w': 0.5}
        options = {'c1': 1.0/2, 'c2': 1.0/2, 'w': 0.5}

        # 初始化优化器
        optimizer = ps.single.GlobalBestPSO(
            n_particles=particle_num,
            dimensions=dim,
            options=options,
            bounds=bounds,
            init_pos = particles
        )

        start_time = time.perf_counter()
        # 优化
        best_cost, best_pos = optimizer.optimize(cost_function_partial, iters=1)
        end_time = time.perf_counter()

        best_par=np.ceil(best_pos[-2])
        best_ser = np.ceil(best_pos[-1])

        param_weight = np.ones(len(self.query_instance.dataset.data_drop_fillna.columns))
        colnames = list(self.query_instance.dataset.data_drop_fillna.columns)
        loss_conf = {
            'reg_weight': 10.0,
            'volt_weight': 10,
            'cap_weight': 6,
            'vol_weight': 6,
            'price_weight': 2,#less important for price data missing
            'ESR_weight': 6/25,
            'ripple_weight': 0.1/25,
        }
        if 'Rated Voltage /V' in colnames:
            idx_voltage = colnames.index('Rated Voltage /V')
            param_weight[idx_voltage] = loss_conf['volt_weight']
        if 'Rated Capacitance /uF' in colnames:
            idx_cap = colnames.index('Rated Capacitance /uF')
            param_weight[idx_cap] = loss_conf['cap_weight']
        if 'Final_Price' in colnames:
            idx_price = colnames.index('Final_Price')
            param_weight[idx_price] = loss_conf['price_weight']
        param_weight[2:7] = loss_conf['vol_weight']
        param_weight[-50:-25] = loss_conf['ESR_weight']
        param_weight[-25:] = loss_conf['ripple_weight']



        real_component_config={'top_k':20,
                               'voltage_constraint':np.max(design_target['V_t']['V_volt'])/best_ser,
                               'capacitance_constraint':design_target['C_uF']/best_par,
                               'param_weight':param_weight}

        best_z_inverse = self.query_instance.find_component_from_latent_space(
            torch.tensor(best_pos[0:-2]).float().unsqueeze(0),real_component_config,mixed_transform=True)
        best_z_inverse_virtual = self.query_instance.find_component_from_latent_space(
            torch.tensor(best_pos[0:-2]).float().unsqueeze(0),real_component_config=None,mixed_transform=True)

        particle = np.zeros((1, 2))
        particle[:, -2] = np.ceil(best_par)
        particle[:, -1] = np.ceil(best_ser)
        cost_values = cost_function_partial(particle, gs_mode=True, designate_dataset=best_z_inverse)
        cost_values_virtual = cost_function_partial(particle, gs_mode=True, designate_dataset=best_z_inverse_virtual)

        opt_min_idx=np.argmin(cost_values)
        best_loss,best_volume=self.cal_loss_vol(best_z_inverse.iloc[[opt_min_idx],:],design_target,best_par,best_ser)
        loss_weight=cost_weight['loss_weight']
        volume_weight = cost_weight['volume_weight']
        score_performance = best_volume * volume_weight + best_loss*loss_weight

        # shape: (num_db,)


        return best_z_inverse
        print('end')





    def grid_search(self, design_target,var_init_config, cost_weight, optimization_config=None):
        if 'particle_num' in var_init_config:
            particle_num = var_init_config['particle_num']
        else:
            particle_num = 10
        if 'var_distribution' in var_init_config:
            init_distribution = var_init_config['var_distribution']
        else:
            init_distribution = ['norm']*18
        init_scale=var_init_config['scale']
        init_shift=var_init_config['shift']

        # init_particles=[]
        # i=0
        # for name,shift,scale in zip(init_distribution,init_shift,init_scale):
        #     if name == 'norm':
        #         sample = np.random.randn(particle_num)
        #     elif name == 'folded_norm':
        #         sample = np.abs(np.random.randn(particle_num))
        #     elif name=='uniform':
        #         sample = np.random.uniform(size=particle_num,low=-1, high=1)
        #     else:
        #         raise ValueError(f"Unsupported distribution {name}")
        #     particle_dim_i=sample*scale+ shift
        #     init_particles.append(particle_dim_i)
        #     i+=1
        # init_particles=np.stack(init_particles,axis=1)
        #
        # particles = init_particles
        # cost=self.cost_function(particles,cost_weight,design_target)

        #only npar$nser
        min_bounds = []
        max_bounds = []
        for name, shift, scale in zip(init_distribution, init_shift, init_scale):
            lower_bound = shift - scale
            upper_bound = shift + scale
            min_bounds.append(lower_bound)
            max_bounds.append(upper_bound)
        bounds = (np.maximum(1.0,np.array(min_bounds)), np.maximum(1.0,np.array(max_bounds)))

        cost_function_partial = partial(
            self.cost_function,
            cost_weight=cost_weight,
            design_target=design_target
        )

        # global_top_k_values=pd.DataFrame(columns=self.query_instance.dataset.data_drop_fillna.columns)
        # top_k=10
        # for n_par in range(bounds[0][-2],bounds[1][-2]):
        #     for n_ser in range(bounds[0][-1], bounds[1][-1]):
        #         particle=np.zeros((1,2))
        #         particle[:, -2]=np.ceil(n_par)
        #         particle[:, -1] = np.ceil(n_ser)
        #         cost_values=cost_function_partial(particle,gs_mode=True)
        #         top_k_idx=np.argsort(cost_values,top_k)
        #         top_k_values=self.query_instance.dataset.data_drop_fillna.iloc[top_k_idx,:]
        #         global_top_k_values=pd.concat([top_k_values,global_top_k_values],axis=0)



        global_top_k_values = pd.DataFrame(columns=self.query_instance.dataset.data_drop_fillna.columns)
        global_top_k_scores = np.array([])  # 保存得分
        global_top_k_indices = np.array([], dtype=int)
        top_k = 10
        for n_par in range(int(bounds[0][-2]), int(bounds[1][-2]) + 1):  # 注意range边界
            for n_ser in range(int(bounds[0][-1]), int(bounds[1][-1]) + 1):
                particle = np.zeros((1, 2))
                particle[:, -2] = np.ceil(n_par)
                particle[:, -1] = np.ceil(n_ser)
                cost_values = cost_function_partial(particle, gs_mode=True)  # shape: (num_db,),calculation

                # 当前参数下top_k
                cur_top_k_idx = np.argsort(cost_values)[:top_k]
                cur_top_k_scores = cost_values[cur_top_k_idx]
                cur_top_k_df = self.query_instance.dataset.data_drop_fillna.iloc[cur_top_k_idx, :].copy()
                cur_top_k_df["n_par"] = n_par
                cur_top_k_df["n_ser"] = n_ser
                cur_top_k_df["score"] = cur_top_k_scores

                # 合并当前top_k与全局top_k
                if global_top_k_values.shape[0] == 0:
                    global_top_k_values = cur_top_k_df
                else:
                    concat_df = pd.concat([global_top_k_values, cur_top_k_df], axis=0, ignore_index=True)
                    # 按score排序
                    concat_df = concat_df.sort_values(by="score", ascending=True)
                    # 只保留top_k
                    global_top_k_values = concat_df.iloc[:top_k, :]


        best_par=global_top_k_values['n_par']
        best_ser =global_top_k_values['n_ser']
        best_score = global_top_k_values['score']


        best_z_inverse = global_top_k_values.drop(['n_par', 'n_ser', 'score'],axis=1)

        best_loss,best_volume=self.cal_loss_vol(best_z_inverse,design_target,best_par.values,best_ser.values) #.iloc[[0],:]

        loss_weight=cost_weight['loss_weight']
        volume_weight = cost_weight['volume_weight']
        score_performance = best_volume * volume_weight + best_loss*loss_weight

        #8.2W, 122cc, 1ser2par, score=144

        return best_z_inverse,(best_score,score_performance ,best_loss,best_volume,best_par,best_ser)
        print('end')

    def sample_uniform_from_gmm(self,gmm, n_samples, random_state=None):
        rng = np.random.default_rng(random_state)

        K = gmm.n_components
        # 1) 每个成分均匀分配样本数（尽量平均）
        counts = np.full(K, n_samples // K, dtype=int)
        counts[: n_samples % K] += 1

        samples = []
        labels = []
        for k, cnt in enumerate(counts):
            if cnt == 0:
                continue
            # 2) 从第 k 个高斯成分采 cnt 个
            xk = rng.multivariate_normal(mean=gmm.means_[k],
                                         cov=gmm.covariances_[k],
                                         size=cnt,
                                         method='eigh')  # 数值更稳
            samples.append(xk)
            labels.append(np.full(cnt, k, dtype=int))

        X = np.vstack(samples)
        comp_ids = np.concatenate(labels)
        # 打乱（可选）
        perm = rng.permutation(n_samples)
        return X[perm], comp_ids[perm]
    def cost_function(self,particle,cost_weight,design_target,gs_mode=False,designate_dataset=None):
        if gs_mode==False:
            particle = np.array(particle)
            component_space=particle[:,:-2]
            n_par=np.ceil(particle[:,-2])
            n_ser=np.ceil(particle[:,-1])
            # find_component_from_latent_space(self,latent_vector,real_component_config=None,mixed_transform=False)
            latent_vector=component_space
            if designate_dataset is None:
                component_params=self.query_instance.find_component_from_latent_space(torch.tensor(latent_vector).float(),mixed_transform=True)
            else:
                component_params = designate_dataset
        else:
            if designate_dataset is None:
                component_params=self.query_instance.dataset.data_drop_fillna_raw
            else:
                component_params=designate_dataset
            #component_params=self.query_instance.dataset.data_drop_fillna
            n_par=np.ones(component_params.shape[0])*np.ceil(particle[:,-2])
            n_ser=np.ones(component_params.shape[0])*np.ceil(particle[:,-1])

        component_capacitance=design_target['C_uF']/n_par
        component_current=design_target['I_f']['Irms_A']/n_par.reshape(-1,1)
        Vmax_component=np.max(design_target['V_t']['V_volt'])/n_ser
        Vav_component=np.mean(design_target['V_t']['V_volt'])/n_ser

        loss_weight=cost_weight['loss_weight']
        volume_weight = cost_weight['volume_weight']
        dist_cap_weight = cost_weight['dist_cap_weight']
        dist_voltage_weight = cost_weight['dist_voltage_weight']
        cons_cap_weight = cost_weight['cons_cap_weight']
        cons_voltage_weight= cost_weight['cons_voltage_weight']
        # loss_weight=100
        # volume_weight=1
        # dist_cap_weight=1
        # dist_voltage_weight=1
        # cons_cap_weight=1000
        # cons_voltage_weight=1000


        loss,volume=self.cal_loss_vol(component_params,design_target,n_par,n_ser)
        score_performance=volume*volume_weight+loss_weight*loss

        capacitance_ratio=(component_params['Rated Capacitance /uF']-component_capacitance)/component_capacitance
        voltage_ratio=(component_params['Rated Voltage /V']-Vmax_component)/Vmax_component
        #test_penalty = (component_params['Type_code'].isin([0.0,1.0,2.0,3.0,4.0,5.0,6.0])).values
        test_penalty = (component_params['Type_code'].isin([0.0,1.0,2.0,3.0,4.0,5.0,6.0,8.0,9.0])).values
        #test_penalty = (component_params['Type_code'].isin([0.0])).values
        score_close_to_target=(np.square(capacitance_ratio)*dist_cap_weight+
                               np.square(voltage_ratio) * dist_voltage_weight
                               +test_penalty*200)

        score_constraints=(-np.minimum(capacitance_ratio,0)*cons_cap_weight+
                               -np.minimum(voltage_ratio,0)* cons_voltage_weight)
        try:
            if self.density_penalty_gmm is not None:
                score_z_dist=self.density_penalty_gmm(latent_vector)
            else:
                score_z_dist=np.zeros_like(latent_vector)
        except:
            score_z_dist=np.zeros_like(score_performance)

        return (score_performance*(1+(np.maximum(score_z_dist,8)-8)/4.0)+score_close_to_target.values+score_constraints.values)

        # test=(score_performance + score_close_to_target.values + score_constraints.values) + (
        #             np.maximum(score_z_dist, 10) * 2)
        # print(score_z_dist[np.argmin(test)])

    def cal_loss_vol(self,component_params,design_target,n_par,n_ser):
        component_current=design_target['I_f']['Irms_A']/n_par.reshape(-1,1)
        freq=self.Hz_deg_list[0]
        temp=self.Hz_deg_list[1]
        esr=component_params.iloc[:,-50:-25]
        target_freq_list = design_target['I_f']['f_Hz']
        target_tempearture=design_target['temperature_deg']
        loss_component=[]
        for i, (idx, row) in enumerate(esr.iterrows()):
            interp_func = RegularGridInterpolator((freq, temp), row.values.reshape(5, 5),bounds_error=False,fill_value=None)
            points = np.stack([target_freq_list, np.full_like(target_freq_list,target_tempearture)], axis=1).tolist()
            esr_row_list=interp_func(points)
            #loss_row=np.array(esr_row_list)*np.square(np.array(component_current[idx,:]))
            loss_row = np.array(esr_row_list) * np.square(np.array(component_current[i, :]))
            loss_component.append(np.sum(loss_row))
        loss_component=np.array(loss_component)/1e3 #to SI unit

        volume_component = np.where(
            np.abs(component_params['Shape_code']) < np.abs(component_params['Shape_code']-1),
            component_params['Diameter /mm'] * component_params['Diameter /mm'] * component_params['Cylinder_length /mm'],
            component_params['Length /mm'] * component_params['Width /mm'] * component_params['Height /mm']
        )
        volume_component=volume_component*1e-9 #SI unit

        loss=loss_component*n_par*n_ser
        volume=volume_component*n_par*n_ser

        return loss, volume
def sine_gen(amplitude, number):
    time = np.linspace(0,2*np.pi,number)
    return amplitude * np.sin(time)

print('test')
# import models
file_abs_path=os.path.dirname(os.path.abspath(__file__))
file_parent_path=os.path.dirname(file_abs_path)
#datasheet_path=os.path.join(file_parent_path,'dataset_embedding/data/datasheets/combine_large_xls/combined_large_excel_v4.3.csv')
# datasheet_path=os.path.join(file_parent_path,'dataset_embedding/data/datasheets/combine_large_xls/combined_large_excel_v4.3.1.csv')
datasheet_path=os.path.join(file_parent_path,'dataset_embedding/data/datasheets/combine_large_xls/combined_large_excel_v5.1.csv')
dataset=MyDataset(csv_file=datasheet_path)



disc_true_table=dataset.data_remain_columns.isin(dataset.category_column)
discrete_feature_list={'true_table':disc_true_table,
                       'dimensions':{'cat_num':[2,10,2],'embedding_dim':[1,2+1,1]}}
hidden_dim=32*8
latent_dim=8
encoder_path=os.path.join(file_parent_path,'dataset_embedding/save/Pre_MWAE/best_model.pth')
model=RCMainMaskedMixedWAE(input_dim=dataset.shape[1], hidden_dim=hidden_dim, latent_dim=latent_dim,disc_dim=4*8,discrete_feature_list=discrete_feature_list,encoder_path=encoder_path)
model_name='Main_RCMWAE'
model_save_path=os.path.join(file_parent_path,'dataset_embedding/save',model_name,'best_model.pth')
model.load_state_dict(torch.load(model_save_path, map_location='cpu'))

# model=MainMaskedMixedWAE(input_dim=dataset.shape[1], hidden_dim=32, latent_dim=8)
# model_name='WAE'
# model_save_path=os.path.join(file_parent_path,'dataset_embedding/save',model_name,'best_model.pth')
# model.load_state_dict(torch.load(model_save_path))



#scaler_save_path=os.path.join(file_parent_path,'dataset_embedding/save','scaler/scaler_params.save')
query_instance=query_datasheet(model=model,dataset=dataset,scaler_save_path=None)
gmm_path=os.path.join(file_parent_path,'dataset_embedding/save/gmm/gmm.joblib')
test_optimizer=optimizer(model,query_instance,gmm_path=gmm_path)
#input
# target_example={
#     'I_f':{'f_Hz':[100,10000],'Irms_A':[10,5]},
#     'V_t':{'t_us':np.linspace(0,0.01*1e6,64).tolist(),'V_volt':300+sine_gen(20,64)},
#     'C_uF':1125,
#     'temperature_deg':25
# }#7.2W128cc，136score
# target_example={
#     'I_f':{'f_Hz':[100,10000],'Irms_A':[20,10]},
#     'V_t':{'t_us':np.linspace(0,0.01*1e6,64).tolist(),'V_volt':100+sine_gen(20,64)},
#     'C_uF':5000,
#     'temperature_deg':25
# }#6W100cc,120score. 8.5W196cc if only type7
target_example={
    'I_f':{'f_Hz':[50,100000],'Irms_A':[5,5]},
    'V_t':{'t_us':np.linspace(0,0.01*1e6,64).tolist(),'V_volt':50+sine_gen(20,64)},
    'C_uF':10100,
    'temperature_deg':50
}#62cc0.58W,36score
Vmax=np.max(target_example['V_t']['V_volt'])
Vav=np.mean(target_example['V_t']['V_volt'])

#latent_dim=16
design_dim=2 #par/ser num
var_init_config={'particle_num':4000,
                 'scale':(np.ones(latent_dim)*2.0).tolist()+[2.5,1.0],
                 'shift':(np.zeros(latent_dim)).tolist()+[2.5,1.0],
                 'var_distribution':['uniform']*(latent_dim+design_dim),
                 'latent_dim':latent_dim+design_dim}
cost_weight = {'loss_weight': 10,
               'volume_weight': 1e6*10/20,
               'dist_cap_weight': 1,
               'dist_voltage_weight': 1,
               'cons_cap_weight': 50000,
               'cons_voltage_weight': 100000}

import time
start_time = time.perf_counter()
test_optimizer.PSO(target_example,var_init_config,cost_weight)
end_time = time.perf_counter()
print(f"执行耗时: {end_time - start_time:.6f} 秒")
start_time = time.perf_counter()
best_z_inverse_gs,best_info=test_optimizer.grid_search(target_example, var_init_config, cost_weight)
end_time = time.perf_counter()
print(f"执行耗时: {end_time - start_time:.6f} 秒")






