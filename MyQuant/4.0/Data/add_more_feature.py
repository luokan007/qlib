"""_summary_
"""

import shutil
import os
from pathlib import Path
import pandas as pd
from ta_lib_feature import TALibFeature
from mydump_bin import DumpDataAll


def _dump_qlib_data(csv_path, qlib_data_path, max_workers=8):
    
    features = f"{qlib_data_path}/features"
    calendars = f"{qlib_data_path}/calendars"

    if os.path.exists(features) and os.path.isdir(features):
        shutil.rmtree(features)

    if os.path.exists(calendars) and os.path.isdir(calendars):
        shutil.rmtree(calendars)

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
    
    today = '2025-01-10' #str(datetime.date.today())
    path = f"{qlib_data_path}/instruments"

    for p in Path(path).iterdir():
        if p.stem == "all":
            continue
        df = pd.read_csv(p, sep="\t", header=None)
        df.sort_values([2, 1, 0], ascending=[False, False, True], inplace=True)  # type: ignore
        latest_data = df[2].max()
        df[2] = df[2].replace(latest_data, today)
        df.to_csv(p, header=False, index=False, sep="\t")

def add_features(data_dir, output_dir,basic_info_path):
    """
    Add technical analysis features to stock data using TALib
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the stock data files
    output_dir : str
        Directory to save the processed files with new features
    """
    ta_feature_generator = TALibFeature(basic_info_path=basic_info_path,time_range=30)

    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        # 使用 shutil.rmtree 高效地移除整个目录树
        shutil.rmtree(output_dir)


    # # 重新创建目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ta_feature_generator.process_directory(data_dir, output_dir)

if __name__ == "__main__":
    # Define input and output directories
    # data_directory = "/home/godlike/project/GoldSparrow/Day_Data/Day_data/Raw"
    # merged_directory = "/home/godlike/project/GoldSparrow/Day_Data/Day_data/Merged_talib"
    # qlib_directory = "/home/godlike/project/GoldSparrow/Day_Data/Day_data/qlib_data"
    # basic_info_path = '/home/godlike/project/GoldSparrow/Day_Data/Day_data/qlib_data/basic_info.csv'

    data_directory = "/root/autodl-tmp/GoldSparrow/Day_data/Raw"
    merged_directory = "/root/autodl-tmp/GoldSparrow/Day_data/Merged_talib"
    qlib_directory = "/root/autodl-tmp/GoldSparrow/Day_data/qlib_data"
    basic_info_path = '/root/autodl-tmp/GoldSparrow/Day_data/qlib_data/basic_info.csv'

    
    # Process the data
    add_features(data_directory, merged_directory,basic_info_path)
    
    ##dump qlib data
    _dump_qlib_data(merged_directory, qlib_directory)