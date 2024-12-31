"""_summary_
"""

import datetime
import shutil
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from ta_lib_feature import TALibFeature
from mydump_bin import DumpDataAll


def _dump_qlib_data(csv_path, qlib_data_path, max_workers=8):
    print("dump qlib data")
    DumpDataAll(
        csv_path= csv_path,
        qlib_dir= qlib_data_path,
        max_workers = max_workers,
        exclude_fields="date,code",
        symbol_field_name="code",
    ).dump()
    shutil.copy(
        f"{qlib_data_path}/calendars/day.txt",
        f"{qlib_data_path}/calendars/day_future.txt",
    )
    _fix_constituents(qlib_data_path)

def _fix_constituents(qlib_data_path):
    
    today = '2024-12-20' #str(datetime.date.today())
    path = f"{qlib_data_path}/instruments"

    for p in Path(path).iterdir():
        if p.stem == "all":
            continue
        df = pd.read_csv(p, sep="\t", header=None)
        df.sort_values([2, 1, 0], ascending=[False, False, True], inplace=True)  # type: ignore
        latest_data = df[2].max()
        df[2] = df[2].replace(latest_data, today)
        df.to_csv(p, header=False, index=False, sep="\t")

def add_features(data_dir, output_dir):
    """
    Add technical analysis features to stock data using TALib
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the stock data files
    output_dir : str
        Directory to save the processed files with new features
    """
    ta_feature_generator = TALibFeature()
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 获取所有CSV文件列表
    files = list(Path(data_dir).glob('*.csv'))

    # 使用tqdm创建进度条
    for file_path in tqdm(files, desc="Processing stocks", unit="file"):
        try:
            # 读取原始数据
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            if df.empty:
                continue
            # 生成新特征
            new_features = ta_feature_generator.generate_single_stock_features(df)

            # 合并特征
            result = pd.concat([df, new_features], axis=1)

            # 保存结果
            output_path = Path(output_dir) / file_path.name
            result.to_csv(output_path)
        except Exception as e:
            print(f"\nError processing {file_path.name}: {str(e)}")

if __name__ == "__main__":
    # Define input and output directories
    data_directory = "/home/godlike/project/GoldSparrow/Day_Data/Day_data/Raw"
    merged_directory = "/home/godlike/project/GoldSparrow/Day_Data/Day_data/Merged_talib"
    qlib_directory = "/home/godlike/project/GoldSparrow/Day_Data/Day_data/qlib_data"
    
    # Process the data
    add_features(data_directory, merged_directory)
    
    ##dump qlib data
    _dump_qlib_data(merged_directory, qlib_directory)