import json
import os
from pathlib import Path
import qlib
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.model import LSTM, ALSTM

class QuantModel:
    def __init__(self, config, work_dir):
        self.config = config
        self.work_dir = work_dir
        self.model = None
        self.dataset = None
        qlib.init(provider_uri=config['provider_uri'], region="cn")
        Path(work_dir).mkdir(parents=True, exist_ok=True)

    def train_evaluate(self):
        self._prepare_data()
        self._initialize_model()
        self.model.fit(dataset=self.dataset)
        pred_series = self.model.predict(dataset=self.dataset)
        pred = pred_series.to_frame("score")
        pred.index = pred.index.set_names(['datetime', 'instrument'])
        pred.to_csv(os.path.join(self.work_dir, "predictions.csv"))
        with open(os.path.join(self.work_dir, "config.json"), 'w') as f:
            json.dump(self.config, f)

    def online_predict(self):
        with open(os.path.join(self.work_dir, "config.json"), 'r') as f:
            config = json.load(f)
        self.config = config
        self._prepare_data()
        self._initialize_model()
        pred_series = self.model.predict(dataset=self.dataset)
        pred = pred_series.to_frame("score")
        pred.index = pred.index.set_names(['datetime', 'instrument'])
        pred.to_csv(os.path.join(self.work_dir, "online_predictions.csv"))

    def _prepare_data(self):
        handler = DataHandlerLP(
            start_time=self.config['train'][0],
            end_time=self.config['test'][1],
            fit_start_time=self.config['fit_start_time'],
            fit_end_time=self.config['fit_end_time'],
            instruments=self.config['pool']
        )
        self.dataset = DatasetH(handler, segments={
            'train': self.config['train'],
            'valid': self.config['valid'],
            'test': self.config['test']
        })

    def _initialize_model(self):
        model_type = self.config['model_type']
        model_params = self.config['model_params']
        if model_type == "lstm":
            self.model = LSTM(**model_params)
        elif model_type == "alstm":
            self.model = ALSTM(**model_params)
        else:
            raise ValueError(f"model_type={model_type} not supported")

# 示例配置字典 - LSTM
config_lstm = {
    'provider_uri': "/root/autodl-tmp/GoldSparrow/Day_data/qlib_data",
    'output_dir': "/root/autodl-tmp/GoldSparrow/Temp_Data",
    'pool': 'csi300',
    'train': ('2008-01-01', '2016-12-31'),
    'valid': ('2017-01-01', '2019-12-31'),
    'test': ('2020-01-01', '2025-01-10'),
    'fit_start_time': '2008-01-01',
    'fit_end_time': '2016-12-31',
    'model_type': 'lstm',
    'model_params': {
        'd_feat': 313,
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0,
        'n_epochs': 50,
        'lr': 0.00001,
        'early_stop': 20,
        'batch_size': 800,
        'metric': "loss",
        'GPU': 0
    }
}

# 示例配置字典 - ALSTM
config_alstm = {
    'provider_uri': "/root/autodl-tmp/GoldSparrow/Day_data/qlib_data",
    'output_dir': "/root/autodl-tmp/GoldSparrow/Temp_Data",
    'pool': 'csi300',
    'train': ('2008-01-01', '2016-12-31'),
    'valid': ('2017-01-01', '2019-12-31'),
    'test': ('2020-01-01', '2025-01-10'),
    'fit_start_time': '2008-01-01',
    'fit_end_time': '2016-12-31',
    'model_type': 'alstm',
    'model_params': {
        'd_feat': 313,
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0,
        'n_epochs': 50,
        'lr': 0.00001,
        'early_stop': 20,
        'batch_size': 800,
        'metric': "loss",
        'loss': "mse",
        'n_jobs': 18,
        'GPU': 0,
        'rnn_type': "GRU"
    }
}

# 使用示例
quant_model_lstm = QuantModel(config_lstm, config_lstm['output_dir'])
quant_model_lstm.train_evaluate()
quant_model_lstm.online_predict()

quant_model_alstm = QuantModel(config_alstm, config_alstm['output_dir'])
quant_model_alstm.train_evaluate()
quant_model_alstm.online_predict()