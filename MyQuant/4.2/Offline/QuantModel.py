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
from qlib.contrib.model.pytorch_lstm import LSTM
from qlib.contrib.model.pytorch_alstm_ts import ALSTM
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import MyAlpha158_DyFeature
from qlib.contrib.data.handler import MyAlphaV4

class QuantModel:
    """_summary_
    """
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
        """_summary_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
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
            evaluation_return = MyEval.from_dataframe(pred, label)
            eval_result = evaluation_return.eval()
        elif model_type == "alstm":
            ## alstm暂时不支持计算
            eval_result = {}

        else:
            raise ValueError(f"model_type={model_type} not supported")
        #print(eval_result)

        # 将评估结果添加到配置中
        self.config['evaluation'] = eval_result

        print("dumping config to ", config_path)
        with open(config_path, 'w') as conf_file_desc:
            json.dump(self.config, conf_file_desc)
        


    def _prepare_data(self):
        """_summary_

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
        """
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

        # handler =MyAlphaV4(instruments=pool,
        #     start_time=start_time,
        #     end_time=end_time,
        #     freq="day",
        #     infer_processors=infer_processors,
        #     learn_processors=learn_processors,
        #     fit_start_time=fit_start_time,
        #     fit_end_time=fit_end_time,     
        #     filter_pipe=filter_rule)


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
    # 'provider_uri': "/root/autodl-tmp/GoldSparrow/Day_data/qlib_data",
    # 'output_dir': "/root/autodl-tmp/GoldSparrow/Temp_Data",
    # 'feature_meta_file': '/root/autodl-tmp/GoldSparrow/Day_data/qlib_data/feature_meta.json',
    'provider_uri': "/home/godlike/project/GoldSparrow/Day_Data/Day_data/qlib_data",
    'output_dir': "/home/godlike/project/GoldSparrow/Temp_Data",
    'feature_meta_file': '/home/godlike/project/GoldSparrow/Day_Data/Day_data/feature_names.json',
    'pool': 'csi300',
    'train': ('2008-01-01', '2020-12-31'),
    'valid': ('2021-01-01', '2022-12-31'),
    'test': ('2023-01-01', '2025-01-30'),
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
     # 'provider_uri': "/root/autodl-tmp/GoldSparrow/Day_data/qlib_data",
    # 'output_dir': "/root/autodl-tmp/GoldSparrow/Temp_Data",
    # 'feature_meta_file': '/root/autodl-tmp/GoldSparrow/Day_data/qlib_data/feature_meta.json',
    'provider_uri': "/home/godlike/project/GoldSparrow/Day_Data/qlib_data",
    'output_dir': "/home/godlike/project/GoldSparrow/Temp_Data",
    'feature_meta_file': '/home/godlike/project/GoldSparrow/Day_Data/feature_names.json',
    'pool': 'csi300',
    'train': ('2008-01-01', '2020-12-31'),
    'valid': ('2021-01-01', '2022-12-31'),
    'test': ('2023-01-01', '2025-02-08'),
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
        'n_jobs': 8,
        'GPU': 0,
        'rnn_type': "GRU"
    }
}

if __name__ == "__main__":
    print("start at ", datetime.now())

    ## init qlib for only one time, otherwise will raise error
    qlib.init(provider_uri=config_alstm['provider_uri'], region="cn")
    #selected_features = ['RANK', 'VWAP0', 'TRIX_48', 'RESI10', 'DAILY_AMOUNT_RATIO', 'RESI5', 'KUP2', 'HIGH0', 'ULTOSC', 'CORD30', 'QTLD5', 'WILLR_6', 'STD60', 'STD20', 'TSF_5', 'CORD5', 'RSQR5', 'STOCHF_k', 'STOCHRSI_k', 'KSFT2', 'STD30', 'STD5', 'QTLU5', 'AMT_VAR_40', 'MA5', 'STD10', 'ROC5', 'KLOW2', 'STR_FACTOR', 'CCI_14', 'WVMA5', 'PBMQR', 'KMID', 'OPEN0', 'RESI20', 'CORR30', 'ADX_14', 'MIN60', 'CORD60', 'ROC_6', 'ADX_28', 'KSFT', 'NATR_28', 'KLEN', 'RESI30', 'AMT_VAR_5', 'KMID2', 'CORD20', 'KUP', 'QTLD10', 'QTLU10', 'CORR60', 'RSQR10', 'RSV5', 'ROC_24', 'ADOSC', 'MA20', 'TURN_MAX_20', 'TSF_10', 'BETA5', 'RANK60', 'STOCHRSI_d', 'AMT_TRIX_40', 'BOP', 'BETA30', 'RSI_6', 'BETA20', 'MIN30', 'IMXD60', 'SIZE', 'STOCHF_d', 'ROC10', 'SUMP5', 'CORR20', 'TURN_MAX_40', 'TRIX_12', 'AMT_SLOPE_5', 'OBV', 'MAX60', 'ROC_12', 'CORR5', 'PSTTM', 'TSF_40', 'MIN10', 'AMT_MAX_5', 'RSQR20', 'LOW0', 'TURN_TSF_5', 'VSTD5', 'TURN_SLOPE_5', 'MIN5', 'CORD10', 'CORR10', 'PETTM', 'KLOW', 'BETA60', 'MFI_6', 'AMT_TSF_5', 'AMT_MIN_40', 'TURN_MIN_5', 'AMT_ROC_40', 'ROC30', 'QTLD30', 'AROON_14_down', 'AD', 'AMT_VAR_20', 'MFI_24', 'RSV60', 'BETA10', 'MA10', 'RANK10', 'AMT_MIN_5', 'VSTD20', 'RSQR60', 'MACD_HIST', 'MA30', 'NATR_14', 'RANK30', 'AMT_TRIX_20', 'TURN_MAX_5', 'CCI_28', 'TURN_MAX_10', 'AMT_TSF_20', 'AMT_ROC_5', 'RSQR30', 'MA60', 'RSV10', 'AMT_RSI_40', 'VSUMN5', 'QTLD20', 'VSUMP5', 'WVMA10', 'VMA5', 'TURN_SLOPE_20', 'ROC60', 'AMT_MAX_40', 'RESI60', 'WILLR_12', 'IMXD30', 'SUMP10', 'VAR_5', 'TRANGE', 'TSF_20', 'MAX5', 'MOM_12', 'MOM_6', 'AMOUNT_LN', 'AMT_MIN_20', 'IMIN60', 'QTLD60', 'VMA60', 'WILLR_24', 'LINEARREG_SLOPE_14', 'SUMN5', 'VSTD60', 'RSI_12', 'TURN_MIN_20', 'VAR_10', 'LINEARREG_SLOPE_5', 'AMT_MAX_20', 'AMT_MAX_10', 'MFI_48', 'SUMD5', 'VAR_40', 'QTLU20', 'TURN_SLOPE_10', 'RSV20', 'WILLR_48', 'VSUMP60', 'TRIX_24', 'SUMN10', 'ROC20', 'MAX30', 'AMT_SLOPE_20', 'TURN_RSI_40', 'AMT_EMA_5', 'AMT_SLOPE_40', 'AMT_TSF_40', 'TURN_TSF_20', 'VMA10', 'TURN_MIN_40', 'AMT_TRIX_10', 'QTLU60', 'AMT_EMA_20', 'MIN20', 'CNTD5', 'RSV30', 'ROC_48', 'VSTD30', 'AMT_MIN_10', 'AMT_VAR_10', 'MAX10', 'TURN_RATE_LN', 'VSTD10', 'IMIN5', 'CMO_28', 'AMT_EMA_40', 'IMIN10', 'MFI_12', 'AMT_TRIX_5', 'TURN_RSI_20', 'TURN_RATE_EMA_5', 'TURN_TSF_10', 'TURN_TSF_40', 'AMT_TSF_10', 'AMT_SLOPE_10', 'WVMA30', 'AMT_RSI_10', 'WVMA60', 'IMAX60', 'CNTD60', 'RANK5', 'VSUMN60', 'CNTN5', 'CNTN20', 'AMT_ROC_10', 'TURN_SLOPE_40', 'VMA30', 'TURN_RATE_EMA_20', 'TURN_RSI_5', 'TURNOVER', 'MACD_SIGNAL', 'WVMA20', 'LINEARREG_SLOPE_28', 'CNTD20', 'MAX20', 'AMT_RSI_5', 'AMT_RSI_20', 'RANK20', 'VAR_20', 'CMO_14', 'TURN_ROC_10', 'TURN_MIN_10', 'AMT_EMA_10', 'VSUMP30', 'TURN_RSI_10', 'RSI_24', 'SUMD10', 'VMA20', 'SUMP60', 'AROON_28_down', 'ATR_14', 'SUMN20', 'CNTP5', 'MOM_48', 'APO', 'SUMN30', 'IMIN20', 'SUMP20', 'SUMN60', 'TURN_ROC_5', 'CNTP60', 'VSUMN20', 'AMT_ROC_20', 'VSUMP10', 'TRIMA_48', 'AROON_28_up', 'SUMP30', 'QTLU30', 'IMXD10', 'CNTD30', 'ATR_28', 'MACD', 'IMIN30', 'CNTN60', 'IMXD20', 'TURN_RATE_EMA_10', 'IMAX5', 'CNTD10', 'VSUMD60', 'VSUMP20', 'MOM_24', 'IMAX10', 'SUMD20', 'IMXD5', 'IMAX20', 'CNTN10', 'CNTP20', 'VSUMN10', 'VSUMD5', 'IMAX30', 'VSUMN30', 'CNTP10', 'TURN_ROC_40', 'CNTP30', 'TEMA_12', 'EMA_10', 'SAR', 'KAMA_24', 'TEMA_48', 'SUMD60', 'VSUMD30', 'EMA_5', 'SUMD30', 'AROON_14_up', 'EMA_20', 'KAMA_48', 'VSUMD20', 'TURN_ROC_20', 'CNTN30']

    ## Alpha 158 + turn/pettm/pbmqr/psttm + STR + RSRS
    # selected_features = ["VWAP0","DAILY_AMOUNT_RATIO","TRIX_48","RESI10","ULTOSC","RESI5","STOCHF_k","QTLD5","KLOW2","STD30",
    #                     "TURNOVER","CORD30","ADOSC","MIN60","STD20","PBMQR","RSQR5","ADX_14","KUP2","RESI20",
    #                     "RSQR10","STD60","QTLU5","CORD20","STR_FACTOR","STOCHRSI_k","STOCHF_d","AD","RSV5","CCI_14",
    #                     "CORR5","WVMA5","WILLR_6","STD10","MA5","RESI30","IMXD60","KSFT","QTLD10","NATR_28",
    #                     "VSTD5","RANK60","STD5","LOW0","KMID","MA20","VSTD20","RSV60","CORR60","KSFT2",
    #                     "RESI60","ROC30","HIGH0","ROC5","CORD60","MFI_24","BETA5","CORD5","MIN30","VMA60",
    #                     "BETA20","VMA5","STOCHRSI_d","OPEN0","ADX_28","WILLR_48","OBV","KMID2","RSV10","CORD10",
    #                     "CORR10","TRANGE","KLEN","BOP","MFI_6","RSV30","BETA10","ROC_6","RSI_6","SUMD5",
    #                     "WILLR_12","MACD_HIST","RSI_12","CORR30","CCI_28","IMXD30","KLOW","PSTTM","norm_RSRS","MA60",
    #                     "MAX5","PETTM","KUP","RSQR30","QTLU10","VMA10","RSQR60","ROC10","ROC_12","MACD_SIGNAL",
    #                     "MA30","RANK30","RSV20","ROC60","VSTD60","MA10","BETA60","NATR_14","SUMN5","TRIX_24",
    #                     "WILLR_24","BETA30","RSQR20","MIN20","MAX60","QTLD20","MAX30","pos_RSRS","APO","CORR20",
    #                     "MIN10","MFI_48","WVMA60","MFI_12","MOM_6","TRIX_12","RANK10","QTLD60","ATR_14","WVMA10",
    #                     "ROC_24","QTLD30","AROON_14_down","MIN5","VMA20","SUMP5","VSUMP5","CMO_14","VSTD30","CNTN60",
    #                     "QTLU20","VSTD10","AROON_28_down","MAX20","RANK20","RANK5","MOM_12","RSI_24","CNTN5","ROC20",
    #                     "IMXD20","IMAX5","MOM_48","AROON_28_up","CMO_28","CNTD10","QTLU60","MAX10","VMA30","VSUMN5",
    #                     "IMIN60","IMXD5","WVMA20","SUMN10","CNTN30","IMXD10","IMIN10","ROC_48","SUMP10","MOM_24",
    #                     "IMIN30","TRIMA_12","TRIMA_48","CNTD30","SUMN60","IMAX60","TEMA_24","WVMA30","CNTN10","VSUMP60",
    #                     "VSUMP20","MACD","VSUMP10","TRIMA_24","TEMA_48","TEMA_12","SUMD10","SUMP60","QTLU30","CNTD20",
    #                     "SUMP30","KAMA_48","CNTN20","VSUMN10","ATR_28","SAR","KAMA_12","SUMN30","CNTP30","EMA_5",
    #                     "CNTP5","VSUMP30","SUMP20","IMIN5","IMAX10","CNTP10","VSUMN30","CNTD5","CNTD60","CNTP60",
    #                     "SUMD30","CNTP20","VSUMN60","IMAX20","VSUMN20","IMAX30","SUMN20","EMA_10","VSUMD5","SUMD60",
    #                     "IMIN20","VSUMD60","EMA_20","KAMA_24","VSUMD10","VSUMD20","SUMD20","AROON_14_up","VSUMD30"]
    
    ## Alpha 158 + turn/pettm/pbmqr/psttm + STR + RSRS + size - VWAP0
    selected_features = ['TURN_MAX_20','AMT_VAR_40','AMT_VAR_5',
                        "VWAP0","DAILY_AMOUNT_RATIO","TRIX_48","RESI10","ULTOSC","RESI5","STOCHF_k","QTLD5","KLOW2","STD30",
                        "CORD30","ADOSC","MIN60","STD20","PBMQR","RSQR5","ADX_14","KUP2","RESI20",
                        "RSQR10","STD60","QTLU5","CORD20","STR_FACTOR","STOCHRSI_k","STOCHF_d","AD","RSV5","CCI_14",
                        "CORR5","WVMA5","WILLR_6","STD10","MA5","RESI30","IMXD60","KSFT","QTLD10","NATR_28",
                        "VSTD5","RANK60","STD5","LOW0","KMID","MA20","VSTD20","RSV60","CORR60","KSFT2",
                        "RESI60","ROC30","HIGH0","ROC5","CORD60","MFI_24","BETA5","CORD5","MIN30","VMA60",
                        "BETA20","VMA5","STOCHRSI_d","OPEN0","ADX_28","WILLR_48","OBV","KMID2","RSV10","CORD10",
                        "CORR10","TRANGE","KLEN","BOP","MFI_6","RSV30","BETA10","ROC_6","RSI_6","SUMD5",
                        "WILLR_12","MACD_HIST","RSI_12","CORR30","CCI_28","IMXD30","KLOW","PSTTM","norm_RSRS","MA60",
                        "MAX5","PETTM","KUP","RSQR30","QTLU10","VMA10","RSQR60","ROC10","ROC_12","MACD_SIGNAL",
                        "MA30","RANK30","RSV20","ROC60","VSTD60","MA10","BETA60","NATR_14","SUMN5","TRIX_24",
                        "WILLR_24","BETA30","RSQR20","MIN20","MAX60","QTLD20","MAX30","pos_RSRS","APO","CORR20",
                        "MIN10","MFI_48","WVMA60","MFI_12","MOM_6","TRIX_12","RANK10","QTLD60","ATR_14","WVMA10",
                        "ROC_24","QTLD30","AROON_14_down","MIN5","VMA20","SUMP5","VSUMP5","CMO_14","VSTD30","CNTN60",
                        "QTLU20","VSTD10","AROON_28_down","MAX20","RANK20","RANK5","MOM_12","RSI_24","CNTN5","ROC20",
                        "IMXD20","IMAX5","MOM_48","AROON_28_up","CMO_28","CNTD10","QTLU60","MAX10","VMA30","VSUMN5",
                        "IMIN60","IMXD5","WVMA20","SUMN10","CNTN30","IMXD10","IMIN10","ROC_48","SUMP10","MOM_24",
                        "IMIN30","TRIMA_12","TRIMA_48","CNTD30","SUMN60","IMAX60","TEMA_24","WVMA30","CNTN10","VSUMP60",
                        "VSUMP20","MACD","VSUMP10","TRIMA_24","TEMA_48","TEMA_12","SUMD10","SUMP60","QTLU30","CNTD20",
                        "SUMP30","KAMA_48","CNTN20","VSUMN10","ATR_28","SAR","KAMA_12","SUMN30","CNTP30","EMA_5",
                        "CNTP5","VSUMP30","SUMP20","IMIN5","IMAX10","CNTP10","VSUMN30","CNTD5","CNTD60","CNTP60",
                        "SUMD30","CNTP20","VSUMN60","IMAX20","VSUMN20","IMAX30","SUMN20","EMA_10","VSUMD5","SUMD60",
                        "IMIN20","VSUMD60","EMA_20","KAMA_24","VSUMD10","VSUMD20","SUMD20","AROON_14_up","VSUMD30"]


    if len(selected_features) > 0:
        quant_model_alstm = QuantModel(config_alstm, config_alstm['output_dir'], selected_features)
    else:
        quant_model_alstm = QuantModel(config_alstm, config_alstm['output_dir'])
    quant_model_alstm.train_evaluate()
    print("done at ", datetime.now())
    # #quant_model_alstm.online_predict()




# 使用示例
# quant_model_lstm = QuantModel(config_lstm, config_lstm['output_dir'])
# quant_model_lstm.train_evaluate()
#quant_model_lstm.online_predict()


## RSRS, 大量去除换手率相关指标，留下四个
    # selected_features = ['norm_RSRS','pos_RSRS',
    #                      'VWAP0', 'TRIX_48', 'RESI10', 'DAILY_AMOUNT_RATIO', 'RESI5', 'KUP2', 'HIGH0', 'ULTOSC', 
    #                      'CORD30', 'QTLD5', 'WILLR_6', 'STD60', 'STD20', 'TSF_5', 'CORD5', 'RSQR5', 'STOCHF_k', 
    #                      'STOCHRSI_k', 'KSFT2', 'STD30', 'STD5', 'QTLU5', 'AMT_VAR_40', 'MA5', 'STD10', 'ROC5', 
    #                      'KLOW2', 'STR_FACTOR', 'CCI_14', 'WVMA5', 'PBMQR', 'KMID', 'OPEN0', 'RESI20', 'CORR30', 
    #                      'ADX_14', 'MIN60', 'CORD60', 'ROC_6', 'ADX_28', 'KSFT', 'NATR_28', 'KLEN', 'RESI30', 
    #                      'AMT_VAR_5', 'KMID2', 'CORD20', 'KUP', 'QTLD10', 'QTLU10', 'CORR60', 'RSQR10', 'RSV5', 
    #                      'ROC_24', 'ADOSC', 'MA20',  'TSF_10', 'BETA5', 'RANK60', 'STOCHRSI_d', 
    #                      'AMT_TRIX_40', 'BOP', 'BETA30', 'RSI_6', 'BETA20', 'MIN30', 'IMXD60', 'SIZE', 'STOCHF_d', 
    #                      'ROC10', 'SUMP5', 'CORR20',  'TRIX_12', 'AMT_SLOPE_5', 'OBV', 'MAX60', 'ROC_12',
    #                      'CORR5', 'PSTTM', 'TSF_40', 'MIN10', 'AMT_MAX_5', 'RSQR20', 'LOW0',  'VSTD5', 
    #                       'MIN5', 'CORD10', 'CORR10', 'PETTM', 'KLOW', 'BETA60', 'MFI_6', 'AMT_TSF_5', 
    #                      'AMT_MIN_40',  'AMT_ROC_40', 'ROC30', 'QTLD30', 'AROON_14_down', 'AD', 'AMT_VAR_20', 
    #                      'MFI_24', 'RSV60', 'BETA10', 'MA10', 'RANK10', 'AMT_MIN_5', 'VSTD20', 'RSQR60', 'MACD_HIST', 'MA30',
    #                      'NATR_14', 'RANK30', 'AMT_TRIX_20',  'CCI_28',  'AMT_TSF_20', 'AMT_ROC_5', 
    #                      'RSQR30', 'MA60', 'RSV10', 'AMT_RSI_40', 'VSUMN5', 'QTLD20', 'VSUMP5', 'WVMA10', 'VMA5',  
    #                      'ROC60', 'AMT_MAX_40', 'RESI60', 'WILLR_12', 'IMXD30', 'SUMP10', 'VAR_5', 'TRANGE', 'TSF_20', 'MAX5', 'MOM_12', 
    #                      'MOM_6', 'AMOUNT_LN', 'AMT_MIN_20', 'IMIN60', 'QTLD60', 'VMA60', 'WILLR_24', 'LINEARREG_SLOPE_14', 'SUMN5', 
    #                      'VSTD60', 'RSI_12', 'TURN_MIN_20', 'TURN_MAX_20', 'TURN_RATE_EMA_20','VAR_10', 'LINEARREG_SLOPE_5', 'AMT_MAX_20', 'AMT_MAX_10', 'MFI_48', 'SUMD5', 
    #                      'VAR_40', 'QTLU20',  'RSV20', 'WILLR_48', 'VSUMP60', 'TRIX_24', 'SUMN10', 'ROC20', 'MAX30',
    #                      'AMT_SLOPE_20',  'AMT_EMA_5', 'AMT_SLOPE_40', 'AMT_TSF_40',  'VMA10', 
    #                      'AMT_TRIX_10', 'QTLU60', 'AMT_EMA_20', 'MIN20', 'CNTD5', 'RSV30', 'ROC_48', 'VSTD30', 'AMT_MIN_10', 'AMT_VAR_10', 
    #                      'MAX10', 'TURN_RATE_LN', 'VSTD10', 'IMIN5', 'CMO_28', 'AMT_EMA_40', 'IMIN10', 'MFI_12', 'AMT_TRIX_5', 'AMT_TSF_10', 'AMT_SLOPE_10', 'WVMA30', 'AMT_RSI_10', 'WVMA60', 'IMAX60', 
    #                      'CNTD60', 'RANK5', 'VSUMN60', 'CNTN5', 'CNTN20', 'AMT_ROC_10',  'VMA30',  
    #                       'MACD_SIGNAL', 'WVMA20', 'LINEARREG_SLOPE_28', 'CNTD20', 'MAX20', 'AMT_RSI_5', 'AMT_RSI_20', 'RANK20', 'VAR_20',
    #                      'CMO_14',   'AMT_EMA_10', 'VSUMP30',  'RSI_24', 'SUMD10', 'VMA20', 'SUMP60', 'AROON_28_down',
    #                      'ATR_14', 'SUMN20', 'CNTP5', 'MOM_48', 'APO', 'SUMN30', 'IMIN20', 'SUMP20', 'SUMN60',  'CNTP60', 'VSUMN20', 'AMT_ROC_20',
    #                      'VSUMP10', 'TRIMA_48', 'AROON_28_up', 'SUMP30', 'QTLU30', 'IMXD10', 'CNTD30', 'ATR_28', 'MACD', 'IMIN30', 'CNTN60', 'IMXD20', 
    #                      'IMAX5', 'CNTD10', 'VSUMD60', 'VSUMP20', 'MOM_24', 'IMAX10', 'SUMD20', 'IMXD5', 'IMAX20', 'CNTN10', 'CNTP20', 'VSUMN10', 'VSUMD5', 
    #                      'IMAX30', 'VSUMN30', 'CNTP10',  'CNTP30', 'TEMA_12', 'EMA_10', 'SAR', 'KAMA_24', 'TEMA_48', 'SUMD60', 'VSUMD30', 'EMA_5', 
    #                      'SUMD30', 'AROON_14_up',  'KAMA_48', 'VSUMD20',  'CNTN30']
    
    ## RSRS 去除TSF和schotastic，
    # selected_features = ['norm_RSRS','pos_RSRS',
    #                      'VWAP0', 'TRIX_48', 'RESI10', 'DAILY_AMOUNT_RATIO', 'RESI5', 'KUP2', 
    #                      'HIGH0', 'ULTOSC', 'CORD30', 'QTLD5', 'WILLR_6', 'STD60', 'STD20',  
    #                      'CORD5', 'RSQR5',    'STD30', 'STD5', 'QTLU5',
    #                      'AMT_VAR_40', 'MA5', 'STD10', 'ROC5', 'KLOW2', 'STR_FACTOR', 'CCI_14', 'WVMA5',
    #                      'PBMQR', 'KMID', 'OPEN0', 'RESI20', 'CORR30', 'ADX_14', 'MIN60', 'CORD60', 'ROC_6', 
    #                      'ADX_28',  'NATR_28', 'KLEN', 'RESI30', 'AMT_VAR_5', 'KMID2', 'CORD20', 'KUP', 
    #                      'QTLD10', 'QTLU10', 'CORR60', 'RSQR10', 'RSV5', 'ROC_24', 'ADOSC', 'MA20', 'TURN_MAX_20',
    #                       'BETA5', 'RANK60',  'AMT_TRIX_40', 'BOP', 'BETA30', 'RSI_6', 'BETA20', 
    #                      'MIN30', 'IMXD60', 'SIZE',  'ROC10', 'SUMP5', 'CORR20', 'TURN_MAX_40', 'TRIX_12', 
    #                      'AMT_SLOPE_5', 'OBV', 'MAX60', 'ROC_12', 'CORR5', 'PSTTM',  'MIN10', 'AMT_MAX_5', 'RSQR20', 
    #                      'LOW0',  'VSTD5', 'TURN_SLOPE_5', 'MIN5', 'CORD10', 'CORR10', 'PETTM', 'KLOW', 'BETA60', 
    #                      'MFI_6',  'AMT_MIN_40', 'TURN_MIN_5', 'AMT_ROC_40', 'ROC30', 'QTLD30', 'AROON_14_down', 'AD', 
    #                      'AMT_VAR_20', 'MFI_24', 'RSV60', 'BETA10', 'MA10', 'RANK10', 'AMT_MIN_5', 'VSTD20', 'RSQR60', 'MACD_HIST', 
    #                      'MA30', 'NATR_14', 'RANK30', 'AMT_TRIX_20', 'TURN_MAX_5', 'CCI_28', 'TURN_MAX_10',  'AMT_ROC_5', 
    #                      'RSQR30', 'MA60', 'RSV10', 'AMT_RSI_40', 'VSUMN5', 'QTLD20', 'VSUMP5', 'WVMA10', 'VMA5', 'TURN_SLOPE_20', 
    #                      'ROC60', 'AMT_MAX_40', 'RESI60', 'WILLR_12', 'IMXD30', 'SUMP10', 'VAR_5', 'TRANGE',  'MAX5', 'MOM_12',
    #                      'MOM_6', 'AMOUNT_LN', 'AMT_MIN_20', 'IMIN60', 'QTLD60', 'VMA60', 'WILLR_24', 'LINEARREG_SLOPE_14', 'SUMN5', 
    #                      'VSTD60', 'RSI_12', 'TURN_MIN_20', 'VAR_10', 'LINEARREG_SLOPE_5', 'AMT_MAX_20', 'AMT_MAX_10', 'MFI_48', 'SUMD5', 
    #                      'VAR_40', 'QTLU20', 'TURN_SLOPE_10', 'RSV20', 'WILLR_48', 'VSUMP60', 'TRIX_24', 'SUMN10', 'ROC20', 'MAX30', 
    #                      'AMT_SLOPE_20', 'TURN_RSI_40', 'AMT_EMA_5', 'AMT_SLOPE_40',   'VMA10', 'TURN_MIN_40', 
    #                      'AMT_TRIX_10', 'QTLU60', 'AMT_EMA_20', 'MIN20', 'CNTD5', 'RSV30', 'ROC_48', 'VSTD30', 'AMT_MIN_10', 'AMT_VAR_10', 
    #                      'MAX10', 'TURN_RATE_LN', 'VSTD10', 'IMIN5', 'CMO_28', 'AMT_EMA_40', 'IMIN10', 'MFI_12', 'AMT_TRIX_5', 'TURN_RSI_20',
    #                      'TURN_RATE_EMA_5',    'AMT_SLOPE_10', 'WVMA30', 'AMT_RSI_10', 'WVMA60', 'IMAX60', 
    #                      'CNTD60', 'RANK5', 'VSUMN60', 'CNTN5', 'CNTN20', 'AMT_ROC_10', 'TURN_SLOPE_40', 'VMA30', 'TURN_RATE_EMA_20', 'TURN_RSI_5',
    #                      'TURNOVER', 'MACD_SIGNAL', 'WVMA20', 'LINEARREG_SLOPE_28', 'CNTD20', 'MAX20', 'AMT_RSI_5', 'AMT_RSI_20', 'RANK20', 
    #                      'VAR_20', 'CMO_14', 'TURN_ROC_10', 'TURN_MIN_10', 'AMT_EMA_10', 'VSUMP30', 'TURN_RSI_10', 'RSI_24', 'SUMD10', 'VMA20',
    #                      'SUMP60', 'AROON_28_down', 'ATR_14', 'SUMN20', 'CNTP5', 'MOM_48', 'APO', 'SUMN30', 'IMIN20', 'SUMP20', 'SUMN60', 'TURN_ROC_5', 
    #                      'CNTP60', 'VSUMN20', 'AMT_ROC_20', 'VSUMP10', 'TRIMA_48', 'AROON_28_up', 'SUMP30', 'QTLU30', 'IMXD10', 'CNTD30', 'ATR_28', 'MACD', 
    #                      'IMIN30', 'CNTN60', 'IMXD20', 'TURN_RATE_EMA_10', 'IMAX5', 'CNTD10', 'VSUMD60', 'VSUMP20', 'MOM_24', 'IMAX10', 'SUMD20', 'IMXD5', 
    #                      'IMAX20', 'CNTN10', 'CNTP20', 'VSUMN10', 'VSUMD5', 'IMAX30', 'VSUMN30', 'CNTP10', 'TURN_ROC_40', 'CNTP30', 'TEMA_12', 'EMA_10', 'SAR', 
    #                      'KAMA_24', 'TEMA_48', 'SUMD60', 'VSUMD30', 'EMA_5', 'SUMD30', 'AROON_14_up', 'EMA_20', 'KAMA_48', 'VSUMD20', 'TURN_ROC_20', 'CNTN30']
    
    ## RSRS + 4.0 特征+
