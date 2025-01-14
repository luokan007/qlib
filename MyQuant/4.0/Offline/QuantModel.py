import json
import os
from pathlib import Path
from some_module import LSTM, ALSTM  # 假设LSTM和ALSTM是从some_module导入的

class QuantModel:
    def __init__(self, config, work_dir):
        self.config = config
        self.work_dir = work_dir
        self.model = self._initialize_model()

    def _initialize_model(self):
        model_type = self.config.get("model_type")
        if model_type == "lstm":
            return LSTM(loss="mse",
                        d_feat=313,
                        hidden_size=64,
                        num_layers=2,
                        dropout=0,
                        n_epochs=50,
                        lr=0.00001,
                        early_stop=20,
                        batch_size=800,
                        metric="loss",
                        GPU=0)
        elif model_type == "alstm":
            return ALSTM(d_feat=313,
                         hidden_size=64,
                         num_layers=2,
                         dropout=0,
                         n_epochs=50,
                         lr=0.00001,
                         early_stop=20,
                         batch_size=800,
                         metric="loss",
                         loss="mse",
                         n_jobs=18,
                         GPU=0,
                         rnn_type="GRU")
        else:
            raise ValueError(f"model_type={model_type} not supported")

    def train_evaluate(self, dataset):
        # 模型训练, 使用fit方法
        self.model.fit(dataset=dataset)

        # 测试集上执行预测
        pred_series = self.model.predict(dataset=dataset)

        # 转换为DataFrame并添加列名
        pred = pred_series.to_frame("score")

        Path(self.work_dir).mkdir(parents=True, exist_ok=True)

        # 格式化索引和导出
        pred.index = pred.index.set_names(['datetime', 'instrument'])
        pred.to_csv(os.path.join(self.work_dir, "predictions.csv"))

        # 保存配置文件
        with open(os.path.join(self.work_dir, "config.json"), "w") as f:
            json.dump(self.config, f)

    def online_predict(self, dataset):
        # 读取配置文件
        with open(os.path.join(self.work_dir, "config.json"), "r") as f:
            config = json.load(f)

        # 初始化模型
        self.config = config
        self.model = self._initialize_model()

        # 执行预测
        pred_series = self.model.predict(dataset=dataset)

        # 转换为DataFrame并添加列名
        pred = pred_series.to_frame("score")

        # 格式化索引和导出
        pred.index = pred.index.set_names(['datetime', 'instrument'])
        pred.to_csv(os.path.join(self.work_dir, "online_predictions.csv"))