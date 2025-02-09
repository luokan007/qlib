# title: get_baostock_seasonal.py
# updated: 2025.1.23
# change log:
#   - 
#   - 

# 目标：
#   1. 从baostock下载季度数据，具体包括：
#   -  成长能力http://baostock.com/baostock/index.php/%E5%AD%A3%E9%A2%91%E6%88%90%E9%95%BF%E8%83%BD%E5%8A%9B
#   - 偿债能力http://baostock.com/baostock/index.php/%E5%AD%A3%E9%A2%91%E5%81%BF%E5%80%BA%E8%83%BD%E5%8A%9B
#   - 运营能力http://baostock.com/baostock/index.php/%E5%AD%A3%E9%A2%91%E8%BF%90%E8%90%A5%E8%83%BD%E5%8A%9B
#   2. 读取股票列表
#   输入：
#       股票列表文件：{qlib_data}/basic_info.csv   由get_baostock_data.py生成
#   输出：
#       文件夹：Raw_Akshare
#       文件：
#   
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

class SeasonalDataDownloader:
    def __init__(self, stock_list_file: str, output_dir: str, max_workers: int = 8):
        self.stock_list_file = stock_list_file
        self.output_dir = output_dir
        self.max_workers = max_workers
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 读取全量股票代码
        with open(self.stock_list_file, "r") as f:
            self.all_stocks = [line.strip() for line in f if line.strip()]

    def download_profit_data(self, code: str) -> pd.DataFrame:
        # TODO: 下载利润数据
        return pd.DataFrame()

    def download_operation_data(self, code: str) -> pd.DataFrame:
        # TODO: 下载营运数据
        return pd.DataFrame()

    def download_growth_data(self, code: str) -> pd.DataFrame:
        # TODO: 下载成长数据
        return pd.DataFrame()

    def download_seasonal_data_for_stock(self, code: str) -> pd.DataFrame:
        # 分别调用下载函数
        df_profit = self.download_profit_data(code)
        df_operation = self.download_operation_data(code)
        df_growth = self.download_growth_data(code)

        # 合并 DataFrame
        df_merged = df_profit  # 先用利润数据作为基础
        # TODO: 依次合并其他 DataFrame 到 df_merged

        return df_merged

    def run(self):
        tasks = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for code in self.all_stocks:
                tasks.append(executor.submit(self.download_seasonal_data_for_stock, code))

            for future in as_completed(tasks):
                try:
                    df = future.result()
                    if not df.empty:
                        code = df["code"].iloc[0] if "code" in df.columns else "unknown"
                        output_path = os.path.join(self.output_dir, f"{code}.csv")
                        df.to_csv(output_path, index=False)
                except Exception as e:
                    print(f"Download failed: {e}")

def main():
    downloader = SeasonalDataDownloader(
        stock_list_file="/path/to/stock_list.txt",
        output_dir="/path/to/output_dir",
        max_workers=8
    )
    downloader.run()

if __name__ == "__main__":
    main()