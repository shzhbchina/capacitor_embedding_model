import pandas as pd
import numpy as np
from thefuzz import process
from io import StringIO  # 用於在程式碼中模擬 CSV 檔案，方便測試
import os
from tqdm import tqdm
import re

# --- 1. 準備範例數據 ---
# 在實際使用中，您會用 pd.read_csv('檔案路徑.csv') 來讀取
# 這裡我們用 StringIO 來模擬您的兩個 CSV 檔案
cur_path=os.path.abspath(os.path.curdir)
parent_path=os.path.dirname(cur_path)
main_csv_path=os.path.join(parent_path,'datasheets/combine_large_xls/combined_large_excel_v4.2.csv')
sub_csv_path=os.path.join(cur_path,'digikey_capacitors_elec_film.csv')
save_path="combined_large_excel_v4.3.csv"

main_df = pd.read_csv(main_csv_path)
supp_df = pd.read_csv(sub_csv_path)

# --- 2. 模糊匹配 ---

# 建立一個包含所有輔文件中「正確」產品編號的列表，作為我們的搜尋目標
choices = supp_df['mfg_number'].tolist()

def clean_part_number(pn_string):
    """
    使用正則表達式清理產品編號，移除可能的批次號等後綴。
    例如：'ELH828M025AR1(1)' -> 'ELH828M025AR1'
           'ELH828M025AQ3AA' -> 'ELH828M025AQ3' (假設 AA 是批次)
    """
    if not isinstance(pn_string, str):
        return ""
    # 這個正則表達式會匹配並移除結尾的 (數字) 或 兩個連續的大寫字母
    # 您可以根據您的數據特徵調整這個規則
    cleaned_pn = re.sub(r'(\(\d+\)|[A-Z]{2})$', '', pn_string)
    return cleaned_pn

def find_best_match(part_number, choices_list, score_cutoff=85):
    """
    為一個給定的 part_number，從 choices_list 中尋找最相似的匹配項。

    Args:
        part_number (str): 要查詢的產品編號。
        choices_list (list): 供選擇的產品編號列表。
        score_cutoff (int): 相似度分數閾值，低於此分數的匹配將被忽略。

    Returns:
        tuple: (最佳匹配的字串, 相似度分數)，如果沒有好的匹配則返回 (None, None)。
    """
    # process.extractOne 會返回一個包含 (匹配項, 分數) 的 tuple
    cleaned_part_number = clean_part_number(part_number)
    best_match = process.extractOne(cleaned_part_number, choices_list)

    if best_match and best_match[1] >= score_cutoff:
        return best_match
    else:
        return (None, None)


# 對主文件的每個 'Part Number' 應用我們的模糊匹配函式
# .apply() 會將結果（一個 tuple）拆分成兩個新的欄位
tqdm.pandas(desc="正在進行模糊匹配...")
main_df[['Best_Match_Supp', 'Match_Score']] = main_df['KEMET Part Number'].progress_apply(
    lambda pn: pd.Series(find_best_match(pn, choices, score_cutoff=85))
)

print("\n--- 步驟 2: 模糊匹配後的結果 ---")
print(main_df)

# --- 3. 合併數據 ---

# 現在我們可以使用「最佳匹配」的結果，與輔文件進行標準的左合併 (left merge)
# 這樣可以將價格和數量資訊帶入主文件
merged_df = pd.merge(
    main_df,
    supp_df,
    how='left',
    left_on='Best_Match_Supp',  # 主文件的匹配鍵
    right_on='mfg_number'  # 輔文件的原始鍵
)

# # 為了清晰，我們可以重新命名和整理一下欄位
# merged_df.rename(columns={'Part Number_x': 'Original_PN', 'Part Number_y': 'Matched_PN'}, inplace=True)

# --- 4. 應用價格邏輯 ---

# 使用 np.where 進行條件式計算，非常高效
# 條件: Quantity >= 100
# 如果為 True: Price * 3.5
# 如果為 False: Price
merged_df['Final_Price'] = np.where(
    merged_df['quantity'] >= 80,  # 條件
    merged_df['price'] * 3,  # 條件為真時的值
    merged_df['price']  # 條件為假時的值
)
merged_df_na=merged_df.fillna('',inplace=False)
columns_to_drop = ['Best_Match_Supp','Match_Score','quantity', 'mfg_number','mfg_name','price','description']
# axis=1 表示我們要刪除的是「欄」，inplace=True 表示直接修改
merged_df_na.drop(columns=columns_to_drop, inplace=True)


#merged_df_na.to_csv(save_path, index=False)
# final_df.to_csv("result.csv", index=False)
# print("\n已保存到 result.csv")