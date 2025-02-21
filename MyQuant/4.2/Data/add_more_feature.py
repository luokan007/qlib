"""_summary_
"""

import shutil
import os
import sys
from pathlib import Path
import pandas as pd
from ta_lib_feature import TALibFeatureExt
sys.path.append('/home/godlike/project/qlib/qlib/MyQuant')
from MyUtil.mydump_bin import DumpDataAll

def _dump_qlib_data(csv_path, qlib_data_path, max_workers=8):
    
    features = f"{qlib_data_path}/features"
    calendars = f"{qlib_data_path}/calendars"

    if os.path.exists(features) and os.path.isdir(features):
        shutil.rmtree(features)

    if os.path.exists(calendars) and os.path.isdir(calendars):
        shutil.rmtree(calendars)

    print("dump qlib data")
    DumpDataAll(
        csv_path=csv_path,
        qlib_dir=qlib_data_path,
        max_workers=max_workers,
        exclude_fields="date,code",
        symbol_field_name="code",
    ).dump()
    shutil.copy(
        f"{qlib_data_path}/calendars/day.txt",
        f"{qlib_data_path}/calendars/day_future.txt",
    )
    _fix_constituents(qlib_data_path)

def _fix_constituents(qlib_data_path):
    
    today = '2025-01-26' #str(datetime.date.today())
    path = f"{qlib_data_path}/instruments"

    for p in Path(path).iterdir():
        if p.stem == "all":
            continue
        df = pd.read_csv(p, sep="\t", header=None)
        df.sort_values([2, 1, 0], ascending=[False, False, True], inplace=True)  # type: ignore
        latest_data = df[2].max()
        df[2] = df[2].replace(latest_data, today)
        df.to_csv(p, header=False, index=False, sep="\t")


if __name__ == "__main__":
    
    work_folder = "/home/godlike/project/GoldSparrow/Day_Data"
    # data_directory = "/root/autodl-tmp/GoldSparrow/Day_data/Raw"  ## 云主机端设置

    ## 本地设置
    data_directory = f"{work_folder}/Raw"
    merged_directory=f"{work_folder}/Merged_talib"
    qlib_directory=f"{work_folder}/qlib_data"
    basic_info_path=f"{work_folder}/qlib_data/basic_info.csv"
    feature_meta_file_path = f"{work_folder}/feature_names.json"
    rsrs_cache_file = f"{work_folder}/rsrs_cache.pkl"
    cpv_feature_config_path = "/home/godlike/project/GoldSparrow/Min_Data/config.json"
    stock_pool_file = None  ##全量数据
    #stock_pool_file = f"{work_folder}/qlib_data/instruments/csi300.txt"  ## 局部数据

    # Process the data
    ta_feature_generator = TALibFeatureExt(basic_info_path=basic_info_path,
                                           time_range=30,
                                           stock_pool_path=stock_pool_file,
                                           rsrs_cache_path=rsrs_cache_file,
                                           cpv_feature_config_path=cpv_feature_config_path,
                                           n_jobs=8)
    
    if os.path.exists(merged_directory) and os.path.isdir(merged_directory):
        # 使用 shutil.rmtree 高效地移除整个目录树
        shutil.rmtree(merged_directory)

    # 重新创建目录
    Path(merged_directory).mkdir(parents=True, exist_ok=True)
    ta_feature_generator.process_directory_incremental(data_directory,
                                                       merged_directory,
                                                       feature_meta_file_path)
    # Dump qlib data after processing
    _dump_qlib_data(merged_directory, qlib_directory)
