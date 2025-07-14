

import pandas as pd
import json
import os
import re
import numpy as np
from scipy.interpolate import interp1d


def_save_json=False
def_current_ESR_feature=False
def_current_ESR_feature_override=True

class predict_electric_performance():

    def __init__(self,df,curve_path):
        self.column_type = None
        self.deg=None
        self.freq=None
        self.df=df
        self.curve_path=curve_path
        self.cap_type_dict=['polymer_capacitor','hybrid_capacitor','electrolytic_capacitor']
        self.one_dimension_curve = None
        self.temps = [0, 40, 80, 120, 160]
        self.freqs = [100, 1000, 10000, 100000, 1000000]
        #self.store_path=store_path
        self.iterate_1d_curve()
        #self.one_dimension_df=None
        self.two_dimension_curve_Ripple = None
    def fit_selected_curve(self,cap_type='',query_curve=''):
        #possible selections for electrolytic caps
        # type='polymer_capacitor'
        # type='hybrid_capacitor'
        # type='electrolytic_capacitor'
        # query_curve='ESR_vs_frequency'
        # query_curve='current_ratio_vs_frequency'

        with open(self.curve_path, 'r', encoding='utf-8') as f:
            current_esr_relation = json.load(f)

        series_read = pd.DataFrame(current_esr_relation[cap_type][query_curve])
        x = series_read.iloc[:, 0].values
        y = series_read.iloc[:, 1].values

        if len(x) == 1:
            # 伪造一个点
            x = np.append(x, x[0] + 1e-6)
            y = np.append(y, y[0])
        fitted_function = interp1d(x, y, kind='linear', fill_value="extrapolate")

        return fitted_function

    def iterate_1d_curve(self):
        #calculate 1d fitting ESR/current_ripple vs
        one_dimension_curve={}
        for cap_type in self.cap_type_dict:
            one_dimension_curve[cap_type] = {}
            for cap_property in ['ESR','current_ratio']:
                append_temp='_vs_temperature'
                append_freq='_vs_frequency'
                property_freq = self.fit_selected_curve(cap_type, cap_property+append_freq)
                property_temp = self.fit_selected_curve(cap_type, cap_property+append_temp)



                one_dimension_curve[cap_type][cap_property+append_freq]=property_freq(self.freqs)
                one_dimension_curve[cap_type][cap_property+append_temp]=property_temp(self.temps)
        self.one_dimension_curve=one_dimension_curve



    def add_cap_property_field(self,store_path):
        # freq=100000
        # temp=25
        # value=16
        #property='ESR'or"Ripple"

        cap_types=['polymer_capacitor', 'hybrid_capacitor', 'electrolytic_capacitor']
        polymer_type= ['Aluminum Organic Capacitor',
                       'Radial Solid Polymer Aluminum Capacitors',
                       'Surface Mount Solid Polymer Aluminum Capacitors']
        electrolytic_type= [ 'Axial Aluminum Electrolytic Capacitors',
                       'Press-Fit Aluminum Electrolytic Capacitors',
                       'Screw Terminal Aluminum Electrolytic Capacitors',
                       'Snap-In Aluminum Electrolytic Capacitors']
        hybrid_type= ['Hybrid Axial Capacitors',
                      'Hybrid Radial Crown Capacitors',
                      'Surface Mount Hybrid Aluminum Polymer Capacitors',]
        cap_type_dict={'polymer_capacitor':polymer_type,
                       'electrolytic_capacitor':electrolytic_type,
                       'hybrid_capacitor':hybrid_type}
        field_types=['ESR','current_ratio']

        for cap_type in cap_types:
            mask = df['Type'].isin(cap_type_dict[cap_type])
            for field_type in field_types:
                # 对frequency相关的批量添加
                for i, freq in enumerate(self.freqs):
                    df.loc[mask, field_type + '_vs_frequency_' + str(freq)] = \
                    self.one_dimension_curve[cap_type][field_type + '_vs_frequency'][i]
                # 对temperature相关的批量添加
                for i, temp in enumerate(self.temps):
                    df.loc[mask, field_type + '_vs_temperature_' + str(temp)] = \
                    self.one_dimension_curve[cap_type][field_type + '_vs_temperature'][i]

        df.to_excel(store_path)
        #self.one_dimension_df=df
        self.df = df

        print('end')

    def value_freq_temp(self,col,dict):
        #parent_dict[col]=dict={'deg','Hz','masked_row'}
        property_base_values=self.df[col].iloc[list(~dict['masked_row'])]
        property_base_freq=dict['Hz']
        property_base_temp=dict['deg']

        frequency_range=[100,1000,10000,100000,1000000]
        temperature_range=[0,40,80,120,160]
        ESR_vs_frequency=self.df[[col_2 for col_2 in self.df.columns if col_2.startswith('ESR_vs_frequency')]]
        ESR_vs_temperature = self.df[[col_2 for col_2 in self.df.columns if col_2.startswith('ESR_vs_temperature')]]
        current_ratio_vs_frequency = self.df[[col_2 for col_2 in self.df.columns if col_2.startswith('current_ratio_vs_frequency')]]
        current_ratio_vs_temperature = self.df[[col_2 for col_2 in self.df.columns if col_2.startswith('current_ratio_vs_temperature')]]

        ESR_stack=[]
        current_ratio_stack=[]
        for i,property_base_value in enumerate(property_base_values):
            if not np.isnan(property_base_value):
                fitted_ESR_vs_frequency = interp1d(frequency_range, ESR_vs_frequency.iloc[i].values, kind='linear', fill_value="extrapolate")
                fitted_ESR_vs_temperature = interp1d(temperature_range, ESR_vs_temperature.iloc[i].values, kind='linear', fill_value="extrapolate")
                fitted_current_ratio_vs_frequency = interp1d(frequency_range, current_ratio_vs_frequency.iloc[i].values, kind='linear', fill_value="extrapolate")
                fitted_current_ratio_vs_temperature = interp1d(temperature_range, current_ratio_vs_temperature.iloc[i].values, kind='linear', fill_value="extrapolate")

                freq_grid, temp_grid = np.meshgrid(frequency_range, temperature_range)
                coords = np.column_stack([freq_grid.ravel(), temp_grid.ravel()])

                coords_ESR=[]
                coords_current_ratio=[]
                for j,coord in enumerate(coords):
                    coords_ESR.append(fitted_ESR_vs_frequency(coord[0])*fitted_ESR_vs_temperature(coord[1]) \
                                    /(fitted_ESR_vs_frequency(property_base_freq)*fitted_ESR_vs_temperature(property_base_temp)) \
                                    *property_base_value)
                    coords_current_ratio.append(fitted_current_ratio_vs_frequency(coord[0])*fitted_current_ratio_vs_temperature(coord[1]) \
                                    /(fitted_current_ratio_vs_frequency(property_base_freq)*fitted_current_ratio_vs_temperature(property_base_temp)) \
                                    *property_base_value)

                ESR_stack.append(coords_ESR)
                current_ratio_stack.append(coords_current_ratio)
        ESR_stack_axis0=np.stack(ESR_stack, axis=0)
        current_ratio_stack_axis0=np.stack(current_ratio_stack, axis=0)
        return ESR_stack_axis0,current_ratio_stack_axis0,coords[:,0],coords[:,1]


    def distance_cal(self,freqs,temps,Hz,deg):
        return np.sqrt(np.square(np.log10(freqs)-np.log10(Hz))+np.square((temps-deg)/40))

    def two_dimension_info_write(self,save_path):
        #require one_dimension_curve columns
        df=self.df
        column_names=['ESR','Ripple']

        column_info = {}
        for column_name in column_names:
            column_name_dict={}
            info_cols = [col for col in df.columns if col.startswith(column_name)]

            pattern = r'(-?\d+(?:\.\d+)?)\s*deg.*?(\d+(?:\.\d+)?)\s*([kK]?Hz)'

            for col in info_cols:
                match = re.search(pattern, col)
                if match:
                    deg = float(match.group(1))
                    hz_val = float(match.group(2))
                    unit = match.group(3).lower()
                    # 统一频率单位为Hz
                    if unit == 'khz':
                        hz = hz_val * 1000
                    else:
                        hz = hz_val
                    column_name_dict[col] = {'deg': deg, 'Hz': hz}
                    # calculate mask and matrix
                    masked_row=df[col].isna()
                    column_name_dict[col]['masked_row']=masked_row
                    # calculate vector
                    coords_ESR,coords_current_ratio,freqs,temps=self.value_freq_temp(col,column_name_dict[col])
                    column_name_dict[col]['coords_ESR'] =coords_ESR
                    column_name_dict[col]['coords_current_ratio'] =coords_current_ratio
                    column_name_dict[col]['freqs'] = freqs
                    column_name_dict[col]['temps'] =temps
                else:
                    print(f'未能识别格式: {col}')
            column_info[column_name] = column_name_dict

        #make up ESR/current_ratio
        weight={}
        large_weight=1000
        for column_type in column_names:
            weight[column_type]={}
            for column_name in column_info[column_type]:
                weight[column_type][column_name] = {}
                deg=column_info[column_type][column_name]['deg']
                Hz=column_info[column_type][column_name]['Hz']
                freqs=column_info[column_type][column_name]['freqs']
                temps=column_info[column_type][column_name]['temps']

                weight[column_type][column_name]['distance']=self.distance_cal(freqs,temps,Hz,deg)
                weight[column_type][column_name]['weight'] = np.divide(
                    1.0,
                    weight[column_type][column_name]['distance'],
                    out=np.full_like(weight[column_type][column_name]['distance'], large_weight, dtype=float),
                    where=weight[column_type][column_name]['distance'] != 0
                )

        #total weight in denominator, rows(5999)*weight(25)
        row_total_weight={}
        for column_type in column_names:
            row_total_weight[column_type]=[]
            for column_name in column_info[column_type]:
                cur_unmasked_row=~column_info[column_type][column_name]['masked_row'].values
                cur_weight=weight[column_type][column_name]['weight']

                row_total_weight[column_type].append(cur_unmasked_row[:,None]*cur_weight[None,:])
            row_total_weight[column_type]=np.sum(row_total_weight[column_type],axis=0)

        # updated ESR/current_ratio
        ESR_current_vector={}
        for column_type in column_names:
            ESR_current_vector[column_type]=[]
            for column_name in column_info[column_type]:
                cur_ESR_current=column_info[column_type][column_name]['coords_ESR']
                cur_weight=weight[column_type][column_name]['weight']
                cur_total_weight=row_total_weight[column_type]
                cur_unmasked_row=~column_info[column_type][column_name]['masked_row'].values
                cur_ESR_current_restore=np.zeros((cur_unmasked_row.shape[0],cur_ESR_current.shape[1]))
                cur_ESR_current_restore[cur_unmasked_row]=cur_ESR_current

                #[row,1,weight]*([1,1,weight]*[row,1,1])/[row,1,weight]
                cur_ESR_current_stack=cur_ESR_current_restore*(cur_weight[None,:]*cur_unmasked_row[:,None])/cur_total_weight
                ESR_current_vector[column_type].append(cur_ESR_current_stack)
            ESR_current_vector[column_type]=np.sum(ESR_current_vector[column_type],axis=0)

        excel_column_names=['ESR','current_ratio']
        for column_type in column_names:
            for column_name in column_info[column_type]:
                freq_name=column_info[column_type][column_name]['freqs']
                temp_name=column_info[column_type][column_name]['temps']
                ESR_value=ESR_current_vector[column_type]

                new_columns = [column_type+ str('_at_') + str(freq_name[i]) + '_Hz_' + str(temp_name[i])+'_deg' for i in range(freq_name.shape[0])]
                for i,col in enumerate(new_columns):
                    df[col]=ESR_value[:,i]

        df.to_excel(save_path)
        return df













##################main
if def_save_json:
    ## convert string category to category number
    # then save mapping to json

    # 1. generate mapping
    df=pd.read_excel('combined_large_excel_v2.xlsx') #v2 doesnot numbering string category field

    df['Type'] = df['Type'].str.strip()  # 去除前后空格
    # print(df['Type'].unique())
    df['Type_code'] = df['Type'].astype('category').cat.codes
    df['Shape_code'] = df['Shape'].astype('category').cat.codes
    df['Manufacturer_code'] = df['Manufacturer'].astype('category').cat.codes
    type_dump = dict(enumerate(df['Type'].astype('category').cat.categories))
    shape_dump = dict(enumerate(df['Shape'].astype('category').cat.categories))
    manufacturer_dump=dict(enumerate(df['Manufacturer'].astype('category').cat.categories))
    # df.to_excel('combined_large_excel_v3.xlsx')
    # 2. save json
    with open('type_dump.json', 'w', encoding='utf-8') as f:
        json.dump(type_dump, f, ensure_ascii=False, indent=4)
    with open('Shape_dump.json', 'w', encoding='utf-8') as f:
        json.dump(shape_dump, f, ensure_ascii=False, indent=4)
    with open('manufacturer_dump.json', 'w', encoding='utf-8') as f:
        json.dump(manufacturer_dump, f, ensure_ascii=False, indent=4)
    # 3. combine mapping jsons into single 'dict.json'

# extract feature current/esr
if def_current_ESR_feature:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.abspath(os.path.join(base_dir, '../../capacitor_relationship/excels'))
    json_path = os.path.join(target_dir, 'convertcsv.json')

    df = pd.read_excel('combined_large_excel_v3.xlsx')
    store_path='combined_large_excel_v3.1_prototype.xlsx'
    test=predict_electric_performance(df,json_path)
    test.add_cap_property_field(store_path)
    #with some modifications, like delete ',' save as utf-8, becomes v3.1


if def_current_ESR_feature_override:
    df = pd.read_excel('combined_large_excel_v3.1.xlsx')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.abspath(os.path.join(base_dir, '../../capacitor_relationship/excels'))
    json_path = os.path.join(target_dir, 'convertcsv.json')
    save_path='combined_large_excel_v4.xlsx'

    test = predict_electric_performance(df, json_path)
    pd=test.two_dimension_info_write(save_path,index=False)






    print('end')

