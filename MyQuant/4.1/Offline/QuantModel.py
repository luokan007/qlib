import json
import os
from pathlib import Path
from datetime import datetime
import qlib
from qlib.data.dataset import DatasetH
from qlib.data.dataset import TSDatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import CSZScoreNorm, DropnaProcessor, RobustZScoreNorm, Fillna,DropnaLabel,CSRankNorm, FilterCol
from qlib.contrib.model.pytorch_lstm import LSTM
from qlib.contrib.model.pytorch_alstm_ts import ALSTM
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import MyAlpha158_DyFeature
from eval_model import MyEval
class QuantModel:
    def __init__(self, config, work_dir, selected_features=None):
        self.config = config
        self.work_dir = work_dir
        self.model = None
        self.dataset = None
        self.selected_features_list = None 
        self.selected_feature_num = None
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        
        if selected_features is not None:
            self.selected_features_list = selected_features
            self.selected_feature_num = len(selected_features)

    def train_evaluate(self):
        self._prepare_data()
        self._initialize_model()
        self.model.fit(dataset=self.dataset)
        pred_series = self.model.predict(dataset=self.dataset)
        pred = pred_series.to_frame("score")
        pred.index = pred.index.set_names(['datetime', 'instrument'])
        
        # 获取当前时间戳
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        model_path = os.path.join(self.work_dir, f"model_{timestamp}.pkl")
        pkl_path = os.path.join(self.work_dir, f"predictions_{timestamp}.pkl")
        csv_path = os.path.join(self.work_dir, f"predictions_{timestamp}.csv")
        config_path = os.path.join(self.work_dir, f"config_{timestamp}.json")
        print(f"configure file : {config_path}")
        
        self.config['model_path'] = model_path
        self.config['prediction_pkl'] = pkl_path
        self.config['prediction_csv'] = csv_path
        
        print("dumping model to ", model_path)
        self.model.to_pickle(model_path)
        pred.to_pickle(pkl_path)
        pred.to_csv(csv_path)
        
        model_type = self.config['model_type']
        
        if model_type == "lstm" or model_type == "gbdt":
            # 提取label文件，计算ic/icir/long precision/short precision
            params = dict(segments="test", col_set="label", data_key=DataHandlerLP.DK_R)
            label = self.dataset.prepare(**params)
            #print(label)
            eval = MyEval.from_dataframe(pred, label)
            eval_result = eval.eval()
        elif model_type == "alstm":
            ## alstm暂时不支持计算
            eval_result = {}
            
        else:
            raise ValueError(f"model_type={model_type} not supported")
        #print(eval_result)

        # 将评估结果添加到配置中
        self.config['evaluation'] = eval_result

        print("dumping config to ", config_path)
        with open(config_path, 'w') as f:
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
        pool = self.config['pool']
        start_time = self.config['train'][0]
        end_time = self.config['test'][1]
        fit_start_time = self.config['train'][0]
        fit_end_time = self.config['train'][1]
        feature_meta_file = self.config['feature_meta_file']

        # 推理处理器，RobustZScoreNorm要算fit_start_time和fit_end_time间的因子均值和方差，
        # 然后因子要减去均值除以标准差就行正则化
        if self.selected_features_list is None:
            infer_processors = [RobustZScoreNorm(fit_start_time=fit_start_time, 
                                                 fit_end_time=fit_end_time, 
                                                  fields_group='feature',
                                                  clip_outlier=True),
                                DropnaProcessor(fields_group='feature')]
        else:
            infer_processors = [FilterCol(fields_group='feature', 
                                          col_list=self.selected_features_list),
                                RobustZScoreNorm(fit_start_time=fit_start_time, 
                                             fit_end_time=fit_end_time, 
                                            fields_group='feature',
                                            clip_outlier=True),
                                Fillna(fields_group='feature')]

        # infer_processors = [RobustZScoreNorm(fit_start_time=fit_start_time, fit_end_time=fit_end_time, 
        #                                       fields_group='feature',
        #                                       clip_outlier=True),DropnaProcessor(fields_group='feature')]
        learn_processors = [DropnaLabel(),CSRankNorm(fields_group='label')]
        filter_rule = None # ExpressionDFilter(rule_expression='EMA($close, 10)<10')
        handler =MyAlpha158_DyFeature(instruments=pool,
            start_time=start_time,
            end_time=end_time,
            freq="day",
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            fit_start_time=fit_start_time,
            fit_end_time=fit_end_time,     
            filter_pipe=filter_rule,
            feature_meta_file=feature_meta_file)

        model_type = self.config['model_type']
        train = self.config['train']
        valid = self.config['valid']
        test = self.config['test']
        
        if self.selected_feature_num is not None:
            if model_type.lower() == "lstm" or model_type.lower() == "alstm":
                self.config["model_params"]["d_feat"] = self.selected_feature_num
            elif model_type.lower() == "gbdt":
                pass # do nothing
            else:
                raise ValueError(f"model_type={model_type} not supported")
        else:
            if model_type.lower() == "lstm" or model_type.lower() == "alstm":
                self.config["model_params"]["d_feat"] = handler.get_feature_count()
            elif model_type.lower() == "gbdt":
                pass
            else:
                raise ValueError(f"model_type={model_type} not supported")
        
        
        
        if model_type.lower() == "lstm":
            self.dataset = DatasetH(handler = handler, segments={"train": train,"valid": valid, "test":test})
        # elif model_type.lower() == "lgbmodel":
        #     self.dataset = DatasetH(handler = handler, segments={"train": train,"valid": valid, "test":test})
        # elif model_type.lower() == "linear":
        #     self.dataset = DatasetH(handler = handler, segments={"train": train,"test":test})
        elif model_type.lower() == "alstm":
            step_len = self.config['model_step_len']
            self.dataset = TSDatasetH(step_len = step_len, handler = handler, segments={"train": train,"valid": valid, "test":test})
        elif model_type.lower() == "gbdt":
            self.dataset = DatasetH(handler = handler, segments={"train": train,"valid": valid, "test":test})
        else:
            raise ValueError(f"model_type={model_type} not supported")
    
    def get_feature_importance(self):
        self._prepare_data()
        self._initialize_model()
        self.model.fit(dataset=self.dataset)
        
        _importance = self.model.get_feature_importance()
        fea_expr, fea_name = self.dataset.handler.get_feature_config() # 获取特征表达式，特征名字
        # 特征名，重要性值的对照字典
        feature_importance = {fea_name[int(i.split('_')[1])]: v for i,v in _importance.items()}
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_importance

    def _initialize_model(self):
        model_type = self.config['model_type']
        model_params = self.config['model_params']
        if model_type == "lstm":
            self.model = LSTM(**model_params)
        elif model_type == "alstm":
            self.model = ALSTM(**model_params)
        elif model_type == "gbdt":
            self.model = LGBModel(**model_params)
        else:
            raise ValueError(f"model_type={model_type} not supported")

# 示例配置字典 - LSTM
config_lstm = {
    'provider_uri': "/root/autodl-tmp/GoldSparrow/Day_data/qlib_data",
    'output_dir': "/root/autodl-tmp/GoldSparrow/Temp_Data",
    'feature_meta_file': '/root/autodl-tmp/GoldSparrow/Day_data/qlib_data/feature_meta.json',
    'pool': 'csi300',
    'train': ('2008-01-01', '2020-12-31'),
    'valid': ('2020-01-01', '2022-12-31'),
    'test': ('2023-01-01', '2025-01-23'),
    'model_type': 'lstm',
    'model_params': {
        'd_feat': 306,
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0,
        'n_epochs': 30,
        'lr': 0.00001,
        'early_stop': 20,
        'batch_size': 800,
        'metric': "loss",
        'GPU': 0
    }
}

# 示例配置字典 - ALSTM
config_alstm = {
    'provider_uri': "/root/autodl-tmp/GoldSparrow/Day_data/qlib_data_csi300",
    'output_dir': "/root/autodl-tmp/GoldSparrow/Temp_Data",
    'feature_meta_file': '/root/autodl-tmp/GoldSparrow/Day_data/qlib_data_all/feature_meta.json',
    'pool': 'csi300',
    'train': ('2008-01-01', '2020-12-31'),
    'valid': ('2021-01-01', '2022-12-31'),
    'test': ('2023-01-01', '2025-01-23'),
    'model_type': 'alstm',
    'model_step_len': 20,
    'model_params': {
        'd_feat': 306,
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0,
        'n_epochs': 20,
        'lr': 0.00001,
        'early_stop': 10,
        'batch_size': 800,
        'metric': "loss",
        'loss': "mse",
        'n_jobs': 18,
        'GPU': 0,
        'rnn_type': "GRU"
    }
}

config_gbdt = {
    'provider_uri': "/root/autodl-tmp/GoldSparrow/Day_data/qlib_data_csi300",
    'output_dir': "/root/autodl-tmp/GoldSparrow/Temp_Data",
    'feature_meta_file': '/root/autodl-tmp/GoldSparrow/Day_data/qlib_data_all/feature_meta.json',
    'pool': 'csi300',
    'train': ('2008-01-01', '2020-12-31'),
    'valid': ('2021-01-01', '2022-12-31'),
    'test': ('2023-01-01', '2025-01-23'),
    'model_type': 'gbdt',
    'model_params': {
        "loss": "mse",
        "colsample_bytree": 0.8879,
        "learning_rate": 0.001, #0.0421,
        "subsample": 0.8789,
        "lambda_l1": 205.6999,
        "lambda_l2": 580.9768,
        "max_depth": 8,
        "num_leaves": 210,
        "num_threads": 20,
        "early_stopping_rounds": 400, # 训练迭代提前停止条件
        "num_boost_round": 2000, # 最大训练迭代次数
    }
}

if __name__ == "__main__":
    print("start at ", datetime.now())

    ## init qlib for only one time, otherwise will raise error
    qlib.init(provider_uri=config_gbdt['provider_uri'], region="cn")

    ##使用GBDT model输出特征的重要性, 筛选特征
    quant_model_gbdt = QuantModel(config_gbdt, config_gbdt['output_dir'])
    feature_importance = quant_model_gbdt.get_feature_importance()
    
    ##output feature importance
    feature_importance_file = os.path.join(config_gbdt['output_dir'], "feature_importance.csv") 
    with open(feature_importance_file, "w") as f:
        for name, imp in feature_importance:
            f.write(f"{name},{imp}\n")
    
    feature_importance_list = [name for name, imp in feature_importance]

    feature_black_list = ['base_RSRS','revise_RSRS','pos_RSRS','norm_RSRS']
    feature_importance_list = [f for f in feature_importance_list if f not in feature_black_list]
    print(f"feature length: {len(feature_importance_list)}")
    #print(feature_importance_list)

    SELECTED_FEATURE_COUNT = 300
    selected_features = feature_importance_list[:SELECTED_FEATURE_COUNT]
    print(f"selected feature list: {selected_features}\n\n")
    print(f"drop feature list: {feature_importance_list[SELECTED_FEATURE_COUNT:]}\n\n")

    quant_model_alstm = QuantModel(config_alstm, config_alstm['output_dir'], selected_features)
    quant_model_alstm.train_evaluate()
    print("done at ", datetime.now())
    # #quant_model_alstm.online_predict()

# 使用示例
# quant_model_lstm = QuantModel(config_lstm, config_lstm['output_dir'])
# quant_model_lstm.train_evaluate()
#quant_model_lstm.online_predict()