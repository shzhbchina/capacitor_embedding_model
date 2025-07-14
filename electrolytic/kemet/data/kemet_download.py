import os
import re
import requests

#fucntion 1: download pdfs from txt index file
def download_pdf(txt_file):
    # 步骤1：读取txt文件并提取所有http(s)链接
    txt_file = 'kemet电容数据手册链接列表.txt'   # 修改为你实际的txt路径

    # 建议保存路径
    save_dir = 'datasheets'
    os.makedirs(save_dir, exist_ok=True)

    # 用于自动提取文件名的正则
    filename_pattern = re.compile(r'\[([^\]]+)\]\((https?://[^\)]+)\)')

    # 统计下载情况
    success, fail = [], []

    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = filename_pattern.search(line)
            if match:
                # 提取名称和URL
                name, url = match.group(1), match.group(2)
                # 自动判断文件后缀
                ext = '.pdf' if '.pdf' in url.lower() else '.pdf'
                # 文件保存名，去掉不合法字符
                clean_name = re.sub(r'[\\/:*?"<>|]', '_', name) + ext
                save_path = os.path.join(save_dir, clean_name)
                # 开始下载
                try:
                    print(f"正在下载: {name} -> {save_path}")
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        with open(save_path, 'wb') as pdf:
                            pdf.write(response.content)
                        print(f"完成: {save_path}")
                        success.append(name)
                    else:
                        print(f"下载失败: {name}，状态码{response.status_code}")
                        fail.append((name, url))
                except Exception as e:
                    print(f"下载异常: {name}，错误信息: {e}")
                    fail.append((name, url))

    print(f"\n下载成功{len(success)}份，失败{len(fail)}份")
    if fail:
        print("未成功的链接：")
        for name, url in fail:
            print(f"{name}: {url}")

    print('end')


#########################################
#function 2:
import tabula
import pandas as pd

def extract_tables_to_excel(pdf_path):
    # 1. 读取PDF中的所有表格（每个表格是一个DataFrame）
    tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)

    # 2. 导出到Excel（每个表格一个sheet，自动命名）
    excel_path = 'A720_series_tables.xlsx'
    with pd.ExcelWriter(excel_path) as writer:
        for i, table in enumerate(tables):
            # 防止无表格内容导致报错
            if not table.empty:
                table.to_excel(writer, sheet_name=f'Table_{i+1}', index=False)

    print(f'已成功提取并保存所有表格到 {excel_path}')


##########################################
#execution
txt_file = 'kemet电容数据手册链接列表.txt'
#download_pdf(txt_file)# 修改为你实际的txt路径

pdf_path='datasheets/A720 Series, +105°C.pdf'
extract_tables_to_excel(pdf_path)

print('end')










