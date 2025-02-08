import json
import os
from pathlib import Path
from datetime import datetime
from eval_model import MyEval
import qlib
from qlib.data.dataset import DatasetH
from qlib.data.dataset import TSDatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import CSZScoreNorm, DropnaProcessor, RobustZScoreNorm, Fillna,DropnaLabel,CSRankNorm, FilterCol
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import MyAlpha158_DyFeature
from qlib.contrib.data.handler import MyAlphaV4

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
                                Fillna(fields_group='feature')]
        else:
            infer_processors = [FilterCol(fields_group='feature', 
                                          col_list=self.selected_features_list),
                                RobustZScoreNorm(fit_start_time=fit_start_time, 
                                             fit_end_time=fit_end_time, 
                                            fields_group='feature',
                                            clip_outlier=True),
                                Fillna(fields_group='feature')]

      
        learn_processors = [DropnaLabel(),CSRankNorm(fields_group='label')]
        filter_rule = None # ExpressionDFilter(rule_expression='EMA($close, 10)<10')
        # handler =MyAlpha158_DyFeature(instruments=pool,
        #     start_time=start_time,
        #     end_time=end_time,
        #     freq="day",
        #     infer_processors=infer_processors,
        #     learn_processors=learn_processors,
        #     fit_start_time=fit_start_time,
        #     fit_end_time=fit_end_time,     
        #     filter_pipe=filter_rule,
        #     feature_meta_file=feature_meta_file)
        
        handler =MyAlphaV4(instruments=pool,
            start_time=start_time,
            end_time=end_time,
            freq="day",
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            fit_start_time=fit_start_time,
            fit_end_time=fit_end_time,     
            filter_pipe=filter_rule)


        model_type = self.config['model_type']
        train = self.config['train']
        valid = self.config['valid']
        test = self.config['test']
        

        if model_type.lower() == "gbdt":
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
        if model_type == "gbdt":
            self.model = LGBModel(**model_params)
        else:
            raise ValueError(f"model_type={model_type} not supported")


config_gbdt = {
     # 'provider_uri': "/root/autodl-tmp/GoldSparrow/Day_data/qlib_data",
    # 'output_dir': "/root/autodl-tmp/GoldSparrow/Temp_Data",
    # 'feature_meta_file': '/root/autodl-tmp/GoldSparrow/Day_data/qlib_data/feature_meta.json',
    'provider_uri': "/home/godlike/project/GoldSparrow/Day_Data/Day_data/qlib_data",
    'output_dir': "/home/godlike/project/GoldSparrow/Temp_Data",
    'feature_meta_file': '/home/godlike/project/GoldSparrow/Day_Data/Day_data/feature_names.json',
    'pool': 'csi300',
    'train': ('2008-01-01', '2021-12-31'),
    'valid': ('2022-01-01', '2022-12-31'),
    'test': ('2023-01-01', '2025-01-30'),
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
    #selected_features = ['RANK', 'VWAP0', 'TRIX_48', 'RESI10', 'DAILY_AMOUNT_RATIO', 'RESI5', 'KUP2', 'HIGH0', 'ULTOSC', 'CORD30', 'QTLD5', 'WILLR_6', 'STD60', 'STD20', 'TSF_5', 'CORD5', 'RSQR5', 'STOCHF_k', 'STOCHRSI_k', 'KSFT2', 'STD30', 'STD5', 'QTLU5', 'AMT_VAR_40', 'MA5', 'STD10', 'ROC5', 'KLOW2', 'STR_FACTOR', 'CCI_14', 'WVMA5', 'PBMQR', 'KMID', 'OPEN0', 'RESI20', 'CORR30', 'ADX_14', 'MIN60', 'CORD60', 'ROC_6', 'ADX_28', 'KSFT', 'NATR_28', 'KLEN', 'RESI30', 'AMT_VAR_5', 'KMID2', 'CORD20', 'KUP', 'QTLD10', 'QTLU10', 'CORR60', 'RSQR10', 'RSV5', 'ROC_24', 'ADOSC', 'MA20', 'TURN_MAX_20', 'TSF_10', 'BETA5', 'RANK60', 'STOCHRSI_d', 'AMT_TRIX_40', 'BOP', 'BETA30', 'RSI_6', 'BETA20', 'MIN30', 'IMXD60', 'SIZE', 'STOCHF_d', 'ROC10', 'SUMP5', 'CORR20', 'TURN_MAX_40', 'TRIX_12', 'AMT_SLOPE_5', 'OBV', 'MAX60', 'ROC_12', 'CORR5', 'PSTTM', 'TSF_40', 'MIN10', 'AMT_MAX_5', 'RSQR20', 'LOW0', 'TURN_TSF_5', 'VSTD5', 'TURN_SLOPE_5', 'MIN5', 'CORD10', 'CORR10', 'PETTM', 'KLOW', 'BETA60', 'MFI_6', 'AMT_TSF_5', 'AMT_MIN_40', 'TURN_MIN_5', 'AMT_ROC_40', 'ROC30', 'QTLD30', 'AROON_14_down', 'AD', 'AMT_VAR_20', 'MFI_24', 'RSV60', 'BETA10', 'MA10', 'RANK10', 'AMT_MIN_5', 'VSTD20', 'RSQR60', 'MACD_HIST', 'MA30', 'NATR_14', 'RANK30', 'AMT_TRIX_20', 'TURN_MAX_5', 'CCI_28', 'TURN_MAX_10', 'AMT_TSF_20', 'AMT_ROC_5', 'RSQR30', 'MA60', 'RSV10', 'AMT_RSI_40', 'VSUMN5', 'QTLD20', 'VSUMP5', 'WVMA10', 'VMA5', 'TURN_SLOPE_20', 'ROC60', 'AMT_MAX_40', 'RESI60', 'WILLR_12', 'IMXD30', 'SUMP10', 'VAR_5', 'TRANGE', 'TSF_20', 'MAX5', 'MOM_12', 'MOM_6', 'AMOUNT_LN', 'AMT_MIN_20', 'IMIN60', 'QTLD60', 'VMA60', 'WILLR_24', 'LINEARREG_SLOPE_14', 'SUMN5', 'VSTD60', 'RSI_12', 'TURN_MIN_20', 'VAR_10', 'LINEARREG_SLOPE_5', 'AMT_MAX_20', 'AMT_MAX_10', 'MFI_48', 'SUMD5', 'VAR_40', 'QTLU20', 'TURN_SLOPE_10', 'RSV20', 'WILLR_48', 'VSUMP60', 'TRIX_24', 'SUMN10', 'ROC20', 'MAX30', 'AMT_SLOPE_20', 'TURN_RSI_40', 'AMT_EMA_5', 'AMT_SLOPE_40', 'AMT_TSF_40', 'TURN_TSF_20', 'VMA10', 'TURN_MIN_40', 'AMT_TRIX_10', 'QTLU60', 'AMT_EMA_20', 'MIN20', 'CNTD5', 'RSV30', 'ROC_48', 'VSTD30', 'AMT_MIN_10', 'AMT_VAR_10', 'MAX10', 'TURN_RATE_LN', 'VSTD10', 'IMIN5', 'CMO_28', 'AMT_EMA_40', 'IMIN10', 'MFI_12', 'AMT_TRIX_5', 'TURN_RSI_20', 'TURN_RATE_EMA_5', 'TURN_TSF_10', 'TURN_TSF_40', 'AMT_TSF_10', 'AMT_SLOPE_10', 'WVMA30', 'AMT_RSI_10', 'WVMA60', 'IMAX60', 'CNTD60', 'RANK5', 'VSUMN60', 'CNTN5', 'CNTN20', 'AMT_ROC_10', 'TURN_SLOPE_40', 'VMA30', 'TURN_RATE_EMA_20', 'TURN_RSI_5', 'TURNOVER', 'MACD_SIGNAL', 'WVMA20', 'LINEARREG_SLOPE_28', 'CNTD20', 'MAX20', 'AMT_RSI_5', 'AMT_RSI_20', 'RANK20', 'VAR_20', 'CMO_14', 'TURN_ROC_10', 'TURN_MIN_10', 'AMT_EMA_10', 'VSUMP30', 'TURN_RSI_10', 'RSI_24', 'SUMD10', 'VMA20', 'SUMP60', 'AROON_28_down', 'ATR_14', 'SUMN20', 'CNTP5', 'MOM_48', 'APO', 'SUMN30', 'IMIN20', 'SUMP20', 'SUMN60', 'TURN_ROC_5', 'CNTP60', 'VSUMN20', 'AMT_ROC_20', 'VSUMP10', 'TRIMA_48', 'AROON_28_up', 'SUMP30', 'QTLU30', 'IMXD10', 'CNTD30', 'ATR_28', 'MACD', 'IMIN30', 'CNTN60', 'IMXD20', 'TURN_RATE_EMA_10', 'IMAX5', 'CNTD10', 'VSUMD60', 'VSUMP20', 'MOM_24', 'IMAX10', 'SUMD20', 'IMXD5', 'IMAX20', 'CNTN10', 'CNTP20', 'VSUMN10', 'VSUMD5', 'IMAX30', 'VSUMN30', 'CNTP10', 'TURN_ROC_40', 'CNTP30', 'TEMA_12', 'EMA_10', 'SAR', 'KAMA_24', 'TEMA_48', 'SUMD60', 'VSUMD30', 'EMA_5', 'SUMD30', 'AROON_14_up', 'EMA_20', 'KAMA_48', 'VSUMD20', 'TURN_ROC_20', 'CNTN30']

    ##使用GBDT model输出特征的重要性, 筛选特征
    quant_model_gbdt = QuantModel(config_gbdt, config_gbdt['output_dir'])
    feature_importance = quant_model_gbdt.get_feature_importance()

    ##output feature importance
    feature_importance_file = os.path.join(config_gbdt['output_dir'], "feature_importance.csv") 
    with open(feature_importance_file, "w") as f:
        for name, imp in feature_importance:
            f.write(f"{name},{imp}\n")

    feature_importance_list = [name for name, imp in feature_importance]
    print("done at ", datetime.now())
    # #quant_model_alstm.online_predict()

