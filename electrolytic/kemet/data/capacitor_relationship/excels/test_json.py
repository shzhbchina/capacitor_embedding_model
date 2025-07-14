#test json application


#test data
import json
import pandas as pd

# --- 第1步: 读取JSON文件 ---
# 使用 'with open' 语法可以确保文件被正确关闭
# 'r' 表示读取模式, 'encoding='utf-8'' 是处理中文等非英文字符的好习惯
with open('convertcsv.json', 'r', encoding='utf-8') as f:
    # json.load() 将文件内容解析为Python字典
    data = json.load(f)

# 此刻, 'data' 变量就是一个包含我们所有数据集的Python字典
# print(data)

# --- 第2步: 选择特定数据集并转换为Pandas DataFrame ---
# 我们的JSON结构非常适合直接转换为DataFrame
# 我们选择 "esr_vs_temperature" 这个数据集作为例子

# 只需将对应的列表传递给 pd.DataFrame() 即可
df_temp = pd.DataFrame(data['esr_vs_temperature'])
df_temp = pd.DataFrame(data['polymer_capacitor']['ESR_vs_frequency'])
df_temp = pd.DataFrame(data['polymer_capacitor']['ESR_vs_temperature'])
df_temp = pd.DataFrame(data['polymer_capacitor']['current_ratio_vs_frequency'])
df_temp = pd.DataFrame(data['polymer_capacitor']['current_ratio_vs_temperature'])
df_temp = data['polymer_capacitor']['lifetime_coefficient']
df_temp = data['polymer_capacitor']['resonance_frequency']

df_temp = pd.DataFrame(data['hybrid_capacitor']['ESR_vs_frequency'])
df_temp = pd.DataFrame(data['hybrid_capacitor']['ESR_vs_temperature'])
df_temp = pd.DataFrame(data['hybrid_capacitor']['current_ratio_vs_frequency'])
df_temp = pd.DataFrame(data['hybrid_capacitor']['current_ratio_vs_temperature'])
df_temp = data['hybrid_capacitor']['lifetime_coefficient']
df_temp = data['hybrid_capacitor']['resonance_frequency']

df_temp = pd.DataFrame(data['electrolytic_capacitor']['ESR_vs_frequency'])
df_temp = pd.DataFrame(data['electrolytic_capacitor']['ESR_vs_temperature'])
df_temp = pd.DataFrame(data['electrolytic_capacitor']['current_ratio_vs_frequency'])
df_temp = pd.DataFrame(data['electrolytic_capacitor']['current_ratio_vs_temperature'])
df_temp = data['electrolytic_capacitor']['lifetime_coefficient']
df_temp = data['electrolytic_capacitor']['resonance_frequency']
# --- 第3步: 查看和使用表格数据 ---
print("--- ESR vs. Temperature 的表格数据 ---")
print(df_temp)

# 您也可以转换其他数据集
df_freq = pd.DataFrame(data['esr_vs_frequency'])
print("\n--- ESR vs. Frequency 的表格数据 ---")
print(df_freq.head()) # .head() 方法默认显示前5行，适合查看大数据集