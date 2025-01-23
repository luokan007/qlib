import os
import json
import pickle
import pandas as pd
from pathlib import Path
import qlib
from qlib.data.dataset import DatasetH
from qlib.data import D
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import RobustZScoreNorm, DropnaProcessor, Fillna,  DropnaLabel, CSRankNorm
from generate_order import GenerateOrder
from qlib.contrib.data.handler import MyAlpha158Ext

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def prepare_data(config):
    provider_uri = config["provider_uri"]
    qlib.init(provider_uri=provider_uri, region="cn")
    
    cal = D.calendar(freq="day")
    latest_date = cal[-1]
    
    ##设置测试数据的时间范围
    test_start_time = latest_date
    test_end_time = latest_date
        
    pool = config['pool']
    start_time = config['train'][0]
    end_time = test_end_time
    fit_start_time = config['train'][0]
    fit_end_time = config['train'][1]

    infer_processors = [RobustZScoreNorm(fit_start_time=fit_start_time, fit_end_time=fit_end_time, 
                                        fields_group='feature',
                                        clip_outlier=True), Fillna(fields_group='feature')]
    learn_processors = [DropnaLabel(), CSRankNorm(fields_group='label')]
    filter_rule = None

    handler = MyAlpha158Ext(
        instruments=pool,
        start_time=start_time,
        end_time=end_time,
        freq="day",
        infer_processors=infer_processors,
        learn_processors=learn_processors,
        fit_start_time=fit_start_time,
        fit_end_time=fit_end_time,
        filter_pipe=filter_rule,
    )

    dataset = DatasetH(handler, 
                       segments={'test': (test_start_time, test_end_time)}
                       )

    return dataset

def main():
    ## 生成订单文件
    basic_info_path = "/root/autodl-tmp/GoldSparrow/Day_data/Meta_Data/stock_basic.csv"
    
    working_dir = "/root/autodl-tmp/GoldSparrow/"
    config_file_name = 'config_20250114191418.json'

    order_folder_name = os.path.join(working_dir, "Order")
    config_folder_name = os.path.join(working_dir, "Temp_Data")
    config_file_path = os.path.join(config_folder_name, config_file_name)
    
    Path(order_folder_name).mkdir(parents=True, exist_ok=True)
    
    config = load_config(config_file_path)
    model_path = config['model_path']
    
    model = load_model(model_path)
    dataset = prepare_data(config)
    pred_series = model.predict(dataset)
    pred_df = pred_series.to_frame("score")
    pred_df.index = pred_df.index.set_names(['datetime', 'instrument'])
    
    order_generator = GenerateOrder(pred_df=pred_df,basic_info_path=basic_info_path,working_dir=order_folder_name)
    order_generator.generate_orders()

if __name__ == "__main__":
    main()