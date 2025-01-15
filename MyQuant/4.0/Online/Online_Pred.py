import os
import json
import pickle
import pandas as pd
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import RobustZScoreNorm, DropnaProcessor, DropnaLabel, CSRankNorm
from GenerateOrder import GenerateOrder
from MyAlpha158Ext import MyAlpha158Ext

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def prepare_data(config):
    pool = config['pool']
    start_time = config['train'][0]
    end_time = config['test'][1]
    fit_start_time = config['train'][0]
    fit_end_time = config['train'][1]

    infer_processors = [RobustZScoreNorm(fit_start_time=fit_start_time, fit_end_time=fit_end_time, 
                                          fields_group='feature',
                                          clip_outlier=True), DropnaProcessor(fields_group='feature')]
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

    dataset = DatasetH(handler, segments={
        'train': config['train'],
        'valid': config['valid'],
        'test': config['test']
    })

    return dataset

def main(config_path):
    config = load_config(config_path)
    model_path = config['model_path']
    
    model = load_model(model_path)
    dataset = prepare_data(config)
    
    test_data = dataset.prepare("test")
    pred_series = model.predict(test_data)
    pred_df = pred_series.to_frame("score")
    pred_df.index = pred_df.index.set_names(['datetime', 'instrument'])
    
    order_generator = GenerateOrder(pred_df)
    order_generator.generate_orders()

if __name__ == "__main__":
    config_path = "config.json"
    main(config_path)