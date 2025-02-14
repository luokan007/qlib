# title: cpv_feature.py
# updated: 2025.2.14
# change log:
#   - 离线处理分钟数据，产生每只股票每一天的cpv因子
#   - 整体架构：
#       - 1) 离线处理，
#              - 生成config文件
#              - 读取分钟数据，生成因子数据,保存到文件，文件路径记录到config文件中
#              - 分钟数据格式
#                   目录结构：
#                       - root_dir
#                           - [year]
#                               - code.csv / 例如  ./2008/sh600381.csv
#                    文件结构
#                         datetime,open,vwap,volume
#                         2008-01-02 09:30,38.5,37.797,31500
#                         2008-01-02 09:31,38.5,37.995,16060
#                         2008-01-02 09:32,38.48,38.173,33140
#                         2008-01-02 09:33,38.3,38.206,14100
#                         2008-01-02 09:34,38.3,38.226,17600
#                         2008-01-02 09:35,38.3,38.225,5600
#                         2008-01-02 09:36,38.27,38.257,28500
#                         2008-01-02 09:37,38.45,38.276,16187
#       - 2) 在线处理，
#              - 读取config文件，返回dict
#               dict：
#                   key： code / 例如sh600031
#                   value： 包含因子数据的dataframe
#                       dataframe格式：
#                           - index: date
#                           - columns: CPV
#       

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

class CPVFeature:
    def __init__(self, config_file, root_dir=None, output_dir=None):
        """
        初始化 CPVFeature 对象：
          离线模式：需要提供 root_dir、output_dir 和 config_file。
          在线模式：只需要提供 config_file。
        """
        self.config_file = Path(config_file).resolve()
        self.root_dir = Path(root_dir).resolve() if root_dir else None
        self.output_dir = Path(output_dir).resolve() if output_dir else None
        self.config_data = {}
        self.merged_output_file = None  # 离线处理后合并的输出文件

    def compute_cpv(self, df, time_range=30):
        """
        计算 CPV 因子：
          1. 输入 df 的索引为 datetime，先过滤出每天时间晚于 14:30 的数据；
          2. 提取日期，将数据按 (code, date_day) 或 (date_day) 分组；
          3. 对每组计算 close 与 volume 的相关系数；
          4. 对每个 code，取最近 time_range 天的相关系数，计算标准差作为 CPV 值。
        """
        # 过滤出每天时间晚于 14:30 的数据
        cutoff = datetime.strptime("14:30", "%H:%M").time()
        df = df[df.index.time > cutoff]
        
        # 提取日期
        df["date_day"] = df.index.date
        
        # 如果 df 中包含 "code" 列，则按 ("code", "date_day") 分组；否则按 "date_day" 分组
        if "code" in df.columns:
            df_ms = df.groupby(["code", "date_day"]).apply(lambda x: x["close"].corr(x["volume"]))
            df_ms = df_ms.reset_index().rename(columns={0: "corr"})
        else:
            df_ms = df.groupby("date_day").apply(lambda x: x["close"].corr(x["volume"]))
            df_ms = df_ms.reset_index().rename(columns={0: "corr"})
            # 若当前处理单只标的，可通过调用方设置 self.current_code
            if hasattr(self, "current_code"):
                df_ms["code"] = self.current_code

        # 定义函数：取每个 code 分组中最近 time_range 天的相关系数，计算标准差作为 CPV
        def last_rows_std(x):
            subset = x.iloc[-time_range:] if len(x) >= time_range else x
            return subset["corr"].std()
        
        if "code" in df_ms.columns:
            cpv_value = df_ms.groupby("code").apply(last_rows_std)
            cpv = pd.DataFrame(cpv_value).reset_index()
            cpv.columns = ["code", "cpv"]
            return cpv
        else:
            std_val = df_ms["corr"].std()
            return pd.DataFrame({"cpv": [std_val]})

    def offline_process(self):
        """
        离线处理流程：
          1. 遍历 root_dir 下所有年份文件夹中的 CSV 文件，
             计算 CPV 因子并为每只股票添加 code 字段，
             将多年的数据合并到一个 DataFrame 中；
          2. 将合并的结果保存到 output_dir 下的一个输出文件中；
          3. 生成 config 文件，记录输出文件夹、数据起止时间、版本信息。
        """
        if self.root_dir is None or self.output_dir is None:
            print("Offline processing requires root_dir and output_dir.")
            return

        all_data = {}  # 用于缓存所有股票的因子数据
        for year_dir in self.root_dir.iterdir():
            if year_dir.is_dir():
                for file_path in year_dir.glob("*.csv"):
                    try:
                        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                        cpv_df = self.compute_cpv(df)
                        # 根据文件名推断代码，例如：sh600381.csv
                        code = file_path.stem
                        
                        if code in all_data:
                            all_data[code] = pd.concat([all_data[code], cpv_df])
                        else:
                            all_data[code] = cpv_df
                        print(f"Processed {file_path} for code {code}.")
                    except Exception as e:
                        print(f"Failed processing {file_path}: {e}")

        if not all_data:
            print("No data processed.")
            return


        
        # 确定数据的起止时间（基于 date 列）
        #start_date = merged_df['date'].min()
        #end_date = merged_df['date'].max()

        # 创建输出目录（如果不存在）
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设定合并后的输出文件路径
        codes = all_data.keys()
        codes.sort()
        
        for code in codes:
            output_file = self.output_dir / f"{code}.csv"
            all_data[code].to_csv(output_file)
            print(f"Saved CPV data to {output_file}")
 
        start_date = min(df['date'] for df in all_data.values())
        end_date = max(df['date'] for df in all_data.values())

        # 保存 config 信息（包含输出目录、起止日期以及版本信息）
        cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.config_data = {
            "output_folder": str(self.output_dir),
            "start_date": str(start_date),
            "end_date": str(end_date),
            "version": "CPV 1.0",
            "generation_time": cur_time
        }
        with self.config_file.open('w', encoding='utf-8') as f:
            json.dump(self.config_data, f, indent=4)
        print(f"Config file saved to {self.config_file}")

    def online_process(self):
        """
        在线处理流程：
          1. 加载 config_file 中的概要信息，获取输出的文件夹路径；
          2. 遍历输出文件夹下的所有 CSV 文件，将多个文件合并为一个 DataFrame；
          3. 返回合并后的 DataFrame。
        """
        if not self.config_file.exists():
            print(f"Config file {self.config_file} does not exist.")
            return None

        with self.config_file.open('r', encoding='utf-8') as f:
            config = json.load(f)

        output_folder = Path(config.get("output_folder", ""))
        if not output_folder.exists():
            print(f"Output folder {output_folder} does not exist.")
            return None

        stock_dfs = {}
        for csv_file in output_folder.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                code = csv_file.stem
                stock_dfs[code] = df
        
                print(f"Loaded CPV data from {csv_file}.")
            except Exception as e:
                print(f"Failed to load CPV data from {csv_file}: {e}")

        if not stock_dfs:
            print("No CPV data files found in the output folder.")
            return None

        return stock_dfs

# 示例使用：

# 离线处理模式（数据处理脚本调用）：
if __name__ == '__main__':
    # 离线处理设置：需要实际的分钟数据目录、因子数据输出目录和 config 文件路径
    root_dir = "/root/autodl-tmp/GoldSparrow/Min_Data/Raw"      # 分钟数据目录，目录结构为：root_dir/[year]/code.csv
    output_dir = "/root/autodl-tmp/GoldSparrow/Min_Data/Processed"    # 因子数据输出目录
    config_file = "/root/autodl-tmp/GoldSparrow/Min_Data/config.json"   # 保存 config 文件的路径

    cpv_feature_offline = CPVFeature(config_file, root_dir, output_dir)
    cpv_feature_offline.offline_process()

    # 在线处理模式（模型学习脚本调用）：只需要 config 文件路径（该路径为固定路径）
    cpv_feature_online = CPVFeature(config_file)
    stock_dfs = cpv_feature_online.online_process()
    print(len(stock_dfs))
    
    codes = list(stock_dfs.keys())
    print(codes[:10])
    print(stock_dfs[codes[0]].head())
