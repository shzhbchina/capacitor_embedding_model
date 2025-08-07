import numpy as np
import os
import torch
from dataset_embedding.models.WassersteinAutoEncoder import WAE
from dataset_embedding.data.dataset import MyDataset
from query.query_datasheet import query_datasheet
from scipy.interpolate import griddata, RegularGridInterpolator
import re
import pyswarms as ps
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from functools import partial

class optimizer():
    def __init__(self,model,query_instance):
        self.model=model
        self.query_instance=query_instance
        column_name = self.query_instance.dataset.data_remain_columns
        self.Hz_deg_list=self.extract_freq_temp(column_name)

        print('end')

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
        particle_num=var_init_config['particle_num']
        init_scale=var_init_config['scale']
        init_distribution=var_init_config['var_distribution']
        init_shift=var_init_config['shift']

        init_particles=[]
        i=0
        for name,shift,scale in zip(init_distribution,init_shift,init_scale):
            if name == 'norm':
                sample = np.random.randn(particle_num)
            elif name == 'folded_norm':
                sample = np.abs(np.random.randn(particle_num))
            elif name=='uniform':
                sample = np.random.uniform(size=particle_num,low=0, high=1)
            else:
                raise ValueError(f"Unsupported distribution {name}")
            particle_dim_i=sample*scale+ shift
            init_particles.append(particle_dim_i)
            i+=1
        init_particles=np.stack(init_particles,axis=1)



        particles=init_particles
        cost=self.cost_function(particles,cost_weight,design_target)

        min_bounds = []
        max_bounds = []
        for name, shift, scale in zip(init_distribution, init_shift, init_scale):
            lower_bound = shift - scale
            upper_bound = shift + scale
            min_bounds.append(lower_bound)
            max_bounds.append(upper_bound)
        bounds = (np.array(min_bounds), np.array(max_bounds))


        cost_function_partial = partial(
            self.cost_function,
            cost_weight=cost_weight,
            design_target=design_target
        )


        dim = 10  # 每维边界
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}  # PSO参数，常用即可

        # 初始化优化器
        optimizer = ps.single.GlobalBestPSO(
            n_particles=1000,
            dimensions=dim,
            options=options,
            bounds=bounds
        )

        # 优化
        best_cost, best_pos = optimizer.optimize(cost_function_partial, iters=20)


        best_par=np.ceil(best_pos[8])
        best_ser = np.ceil(best_pos[9])

        real_component_config={'top_k':10,
                               'voltage_constraint':np.max(design_target['V_t']['V_volt'])/best_ser,
                               'capacitance_constraint':design_target['C_uF']/best_par,
                               'param_weight':np.array([1]*12+[0.1]*50)}

        best_z_inverse = self.query_instance.find_component_from_latent_space(
            torch.tensor(best_pos[0:8]).float().unsqueeze(0),real_component_config)

        best_loss,best_volume=self.cal_loss_vol(best_z_inverse,design_target,best_par,best_ser)



        return best_z_inverse
        print('end')

    def cost_function(self,particle,cost_weight,design_target):
        particle = np.array(particle)
        component_space=particle[:,:8]
        n_par=np.ceil(particle[:,8])
        n_ser=np.ceil(particle[:,9])
        component_params=self.query_instance.find_component_from_latent_space(torch.tensor(component_space).float())
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
        score_close_to_target=(np.square(capacitance_ratio)*dist_cap_weight+
                               np.square(voltage_ratio) * dist_voltage_weight)

        score_constraints=(-np.minimum(capacitance_ratio,0)*cons_cap_weight+
                               -np.minimum(voltage_ratio,0)* cons_voltage_weight)

        return score_performance+score_close_to_target.values+score_constraints.values

    def cal_loss_vol(self,component_params,design_target,n_par,n_ser):
        component_current=design_target['I_f']['Irms_A']/n_par.reshape(-1,1)
        freq=self.Hz_deg_list[0]
        temp=self.Hz_deg_list[1]
        esr=component_params.iloc[:,-50:-25]
        target_freq_list = design_target['I_f']['f_Hz']
        target_tempearture=design_target['temperature_deg']
        loss_component=[]
        for idx,row in esr.iterrows():
            interp_func = RegularGridInterpolator((freq, temp), row.values.reshape(5, 5))
            points = np.stack([target_freq_list, np.full_like(target_freq_list,target_tempearture)], axis=1).tolist()
            esr_row_list=interp_func(points)
            loss_row=np.array(esr_row_list)*np.square(np.array(component_current[idx,:]))
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
datasheet_path=os.path.join(file_parent_path,'dataset_embedding/data/datasheets/combine_large_xls/combined_large_excel_v4.2.csv')
dataset=MyDataset(csv_file=datasheet_path)
model=WAE(input_dim=dataset.shape[1], hidden_dim=32, latent_dim=8)
model_name='WAE'
model_save_path=os.path.join(file_parent_path,'dataset_embedding/save',model_name,'best_model.pth')
model.load_state_dict(torch.load(model_save_path))

scaler_save_path=os.path.join(file_parent_path,'dataset_embedding/save','scaler/scaler_params.save')
query_instance=query_datasheet(model=model,dataset=dataset,scaler_save_path=scaler_save_path)

test_optimizer=optimizer(model,query_instance)
#input
target_example={
    'I_f':{'f_Hz':[100,10000],'Irms_A':[10,5]},
    'V_t':{'t_us':np.linspace(0,0.01*1e6,64).tolist(),'V_volt':300+sine_gen(20,64)},
    'C_uF':1125,
    'temperature_deg':25
}
Vmax=np.max(target_example['V_t']['V_volt'])
Vav=np.mean(target_example['V_t']['V_volt'])

var_init_config={'particle_num':1000,
                 'scale':np.ones(8).tolist()+[2.5,2.5],
                 'shift':np.zeros(8).tolist()+[2.5,2.5],
                 'var_distribution':['norm']*8+['uniform']*2}
cost_weight = {'loss_weight': 10,
               'volume_weight': 1e6*10/20,
               'dist_cap_weight': 1,
               'dist_voltage_weight': 1,
               'cons_cap_weight': 50000,
               'cons_voltage_weight': 100000}
test_optimizer.PSO(target_example,var_init_config,cost_weight)



