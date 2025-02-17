# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from qlib.contrib.data.loader import Alpha158DL, Alpha360DL
from ...data.dataset.handler import DataHandlerLP
from ...data.dataset.processor import Processor
from ...utils import get_callable_kwargs
from ...data.dataset import processor as processor_module
from inspect import getfullargspec
import json


def check_transform_proc(proc_l, fit_start_time, fit_end_time):
    new_l = []
    for p in proc_l:
        if not isinstance(p, Processor):
            klass, pkwargs = get_callable_kwargs(p, processor_module)
            args = getfullargspec(klass).args
            if "fit_start_time" in args and "fit_end_time" in args:
                assert (
                    fit_start_time is not None and fit_end_time is not None
                ), "Make sure `fit_start_time` and `fit_end_time` are not None."
                pkwargs.update(
                    {
                        "fit_start_time": fit_start_time,
                        "fit_end_time": fit_end_time,
                    }
                )
            proc_config = {"class": klass.__name__, "kwargs": pkwargs}
            if isinstance(p, dict) and "module_path" in p:
                proc_config["module_path"] = p["module_path"]
            new_l.append(proc_config)
        else:
            new_l.append(p)
    return new_l


_DEFAULT_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]
_DEFAULT_INFER_PROCESSORS = [
    {"class": "ProcessInf", "kwargs": {}},
    {"class": "ZScoreNorm", "kwargs": {}},
    {"class": "Fillna", "kwargs": {}},
]


class Alpha360(DataHandlerLP):
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=_DEFAULT_INFER_PROCESSORS,
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        filter_pipe=None,
        inst_processors=None,
        **kwargs
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": Alpha360DL.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            learn_processors=learn_processors,
            infer_processors=infer_processors,
            **kwargs
        )

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]


class Alpha360vwap(Alpha360):
    def get_label_config(self):
        return ["Ref($vwap, -2)/Ref($vwap, -1) - 1"], ["LABEL0"]


class Alpha158(DataHandlerLP):
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        **kwargs
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs
        )

    def get_feature_config(self):
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {},
        }
        return Alpha158DL.get_feature_config(conf)

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]


class Alpha158vwap(Alpha158):
    def get_label_config(self):
        return ["Ref($vwap, -2)/Ref($vwap, -1) - 1"], ["LABEL0"]

class MyAlpha158Ext(DataHandlerLP):
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        **kwargs
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs
        )
    def get_feature_config(self):
        fields, names = [],[]
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {},
        }
        fields, names = Alpha158DL.get_feature_config(conf)
        
        fields += ['$turn','$peTTM', '$pbMRQ', '$psTTM']
        names += ['TURNOVER', 'PETTM', 'PBMQR', 'PSTTM']
        ## add baostock basic features
        ## turn / peTTM / pbMRQ / psTTM / pcfNcfTTM / pbMRQ
        
        # fields += ['Ref($turn, 1)/$turn', '$turn', '$peTTM', '$pbMRQ', '$psTTM', '$pcfNcfTTM', '$pbMRQ']
        # names += ['TURNOVER1', 'TURNOVER0', 'PETTM', 'PBMQR', 'PSTTM', 'PCFNCF', 'PBMQR']
        
        # v4.2，totally 309 features: 147（ta-lib) + 158(初始的) + 4（（turn / peTTM / pbMRQ / psTTM））
        fields += [
            '$AD', '$ADOSC', '$ADX_14', '$ADX_28', '$AMOUNT_LN', '$AMT_EMA_10', '$AMT_EMA_20',
            '$AMT_EMA_40', '$AMT_EMA_5', '$AMT_MAX_10', '$AMT_MAX_20', '$AMT_MAX_40', '$AMT_MAX_5',
            '$AMT_MIN_10', '$AMT_MIN_20', '$AMT_MIN_40', '$AMT_MIN_5', '$AMT_ROC_10', '$AMT_ROC_20',
            '$AMT_ROC_40', '$AMT_ROC_5', '$AMT_RSI_10', '$AMT_RSI_20', '$AMT_RSI_40', '$AMT_RSI_5',
            '$AMT_SLOPE_10', '$AMT_SLOPE_20', '$AMT_SLOPE_40', '$AMT_SLOPE_5', '$AMT_TRIX_10', '$AMT_TRIX_20',
            '$AMT_TRIX_40', '$AMT_TRIX_5', '$AMT_TSF_10', '$AMT_TSF_20', '$AMT_TSF_40', '$AMT_TSF_5',
            '$AMT_VAR_10', '$AMT_VAR_20', '$AMT_VAR_40', '$AMT_VAR_5', '$APO', '$AROON_14_down', '$AROON_14_up',
            '$AROON_28_down', '$AROON_28_up', '$ATR_14', '$ATR_28', '$BOP', '$CCI_14', '$CCI_28', '$CMO_14',
            '$CMO_28', '$DAILY_AMOUNT_RATIO', '$EMA_10', '$EMA_20', '$EMA_5', '$KAMA_12', '$KAMA_24',
            '$KAMA_48', '$LINEARREG_SLOPE_14', '$LINEARREG_SLOPE_28', '$LINEARREG_SLOPE_5', '$MACD',
            '$MACD_HIST', '$MACD_SIGNAL', '$MFI_12', '$MFI_24', '$MFI_48', '$MFI_6', '$MOM_12', '$MOM_24',
            '$MOM_48', '$MOM_6', '$NATR_14', '$NATR_28', '$OBV', '$RANK', '$ROC_12', '$ROC_24',
            '$ROC_48', '$ROC_6', '$RSI_12', '$RSI_24', '$RSI_6', '$SAR', '$SIZE', '$STOCHF_d', '$STOCHF_k',
            '$STOCHRSI_d', '$STOCHRSI_k', '$STR_FACTOR', '$TEMA_12', '$TEMA_24', '$TEMA_48', '$TRANGE',
            '$TRIMA_12', '$TRIMA_24', '$TRIMA_48', '$TRIX_12', '$TRIX_24', '$TRIX_48', '$TSF_10',
            '$TSF_20', '$TSF_40', '$TSF_5', '$TURN_MAX_10', '$TURN_MAX_20', '$TURN_MAX_40', '$TURN_MAX_5',
            '$TURN_MIN_10', '$TURN_MIN_20', '$TURN_MIN_40', '$TURN_MIN_5', '$TURN_RATE_EMA_10',
            '$TURN_RATE_EMA_20', '$TURN_RATE_EMA_5', '$TURN_RATE_LN', '$TURN_ROC_10', '$TURN_ROC_20',
            '$TURN_ROC_40', '$TURN_ROC_5', '$TURN_RSI_10', '$TURN_RSI_20', '$TURN_RSI_40', '$TURN_RSI_5',
            '$TURN_SLOPE_10', '$TURN_SLOPE_20', '$TURN_SLOPE_40', '$TURN_SLOPE_5', '$TURN_TSF_10',
            '$TURN_TSF_20', '$TURN_TSF_40', '$TURN_TSF_5', '$ULTOSC', '$VAR_10', '$VAR_20', '$VAR_40',
            '$VAR_5', '$WILLR_12', '$WILLR_24', '$WILLR_48', '$WILLR_6',  '$base_RSRS','$norm_RSRS','$pos_RSRS','$revise_RSRS'
        ]

        names += [
            'AD', 'ADOSC', 'ADX_14', 'ADX_28', 'AMOUNT_LN', 'AMT_EMA_10', 'AMT_EMA_20', 'AMT_EMA_40',
            'AMT_EMA_5', 'AMT_MAX_10', 'AMT_MAX_20', 'AMT_MAX_40', 'AMT_MAX_5', 'AMT_MIN_10', 'AMT_MIN_20',
            'AMT_MIN_40', 'AMT_MIN_5', 'AMT_ROC_10', 'AMT_ROC_20', 'AMT_ROC_40', 'AMT_ROC_5', 'AMT_RSI_10',
            'AMT_RSI_20', 'AMT_RSI_40', 'AMT_RSI_5', 'AMT_SLOPE_10', 'AMT_SLOPE_20', 'AMT_SLOPE_40',
            'AMT_SLOPE_5', 'AMT_TRIX_10', 'AMT_TRIX_20', 'AMT_TRIX_40', 'AMT_TRIX_5', 'AMT_TSF_10',
            'AMT_TSF_20', 'AMT_TSF_40', 'AMT_TSF_5', 'AMT_VAR_10', 'AMT_VAR_20', 'AMT_VAR_40', 'AMT_VAR_5',
            'APO', 'AROON_14_down', 'AROON_14_up', 'AROON_28_down', 'AROON_28_up', 'ATR_14', 'ATR_28',
            'BOP', 'CCI_14', 'CCI_28', 'CMO_14', 'CMO_28', 'DAILY_AMOUNT_RATIO', 'EMA_10', 'EMA_20',
            'EMA_5', 'KAMA_12', 'KAMA_24', 'KAMA_48', 'LINEARREG_SLOPE_14', 'LINEARREG_SLOPE_28',
            'LINEARREG_SLOPE_5', 'MACD', 'MACD_HIST', 'MACD_SIGNAL', 'MFI_12', 'MFI_24', 'MFI_48',
            'MFI_6', 'MOM_12', 'MOM_24', 'MOM_48', 'MOM_6', 'NATR_14', 'NATR_28', 'OBV', 'RANK',
            'ROC_12', 'ROC_24', 'ROC_48', 'ROC_6', 'RSI_12', 'RSI_24', 'RSI_6', 'SAR', 'SIZE',
            'STOCHF_d', 'STOCHF_k', 'STOCHRSI_d', 'STOCHRSI_k', 'STR_FACTOR', 'TEMA_12', 'TEMA_24',
            'TEMA_48', 'TRANGE', 'TRIMA_12', 'TRIMA_24', 'TRIMA_48', 'TRIX_12', 'TRIX_24', 'TRIX_48',
            'TSF_10', 'TSF_20', 'TSF_40', 'TSF_5', 'TURN_MAX_10', 'TURN_MAX_20', 'TURN_MAX_40', 'TURN_MAX_5',
            'TURN_MIN_10', 'TURN_MIN_20', 'TURN_MIN_40', 'TURN_MIN_5', 'TURN_RATE_EMA_10', 'TURN_RATE_EMA_20',
            'TURN_RATE_EMA_5', 'TURN_RATE_LN', 'TURN_ROC_10', 'TURN_ROC_20', 'TURN_ROC_40', 'TURN_ROC_5',
            'TURN_RSI_10', 'TURN_RSI_20', 'TURN_RSI_40', 'TURN_RSI_5', 'TURN_SLOPE_10', 'TURN_SLOPE_20',
            'TURN_SLOPE_40', 'TURN_SLOPE_5', 'TURN_TSF_10', 'TURN_TSF_20', 'TURN_TSF_40', 'TURN_TSF_5',
            'ULTOSC', 'VAR_10', 'VAR_20', 'VAR_40', 'VAR_5', 'WILLR_12', 'WILLR_24', 'WILLR_48',
            'WILLR_6', 'base_RSRS','norm_RSRS','pos_RSRS','revise_RSRS'
        ]
        
        # ## v4.1, totally 305 features: 143(ta-lib) + 158(初始的) + 4（（turn / peTTM / pbMRQ / psTTM））
        # fields += ['$AD',
        #             '$ADOSC',
        #             '$ADX_14',
        #             '$ADX_28',
        #             '$AMOUNT_LN',
        #             '$AMT_EMA_10',
        #             '$AMT_EMA_20',
        #             '$AMT_EMA_40',
        #             '$AMT_EMA_5',
        #             '$AMT_MAX_10',
        #             '$AMT_MAX_20',
        #             '$AMT_MAX_40',
        #             '$AMT_MAX_5',
        #             '$AMT_MIN_10',
        #             '$AMT_MIN_20',
        #             '$AMT_MIN_40',
        #             '$AMT_MIN_5',
        #             '$AMT_ROC_10',
        #             '$AMT_ROC_20',
        #             '$AMT_ROC_40',
        #             '$AMT_ROC_5',
        #             '$AMT_RSI_10',
        #             '$AMT_RSI_20',
        #             '$AMT_RSI_40',
        #             '$AMT_RSI_5',
        #             '$AMT_SLOPE_10',
        #             '$AMT_SLOPE_20',
        #             '$AMT_SLOPE_40',
        #             '$AMT_SLOPE_5',
        #             '$AMT_TRIX_10',
        #             '$AMT_TRIX_20',
        #             '$AMT_TRIX_40',
        #             '$AMT_TRIX_5',
        #             '$AMT_TSF_10',
        #             '$AMT_TSF_20',
        #             '$AMT_TSF_40',
        #             '$AMT_TSF_5',
        #             '$AMT_VAR_10',
        #             '$AMT_VAR_20',
        #             '$AMT_VAR_40',
        #             '$AMT_VAR_5',
        #             '$APO',
        #             '$AROON_14_down',
        #             '$AROON_14_up',
        #             '$AROON_28_down',
        #             '$AROON_28_up',
        #             '$ATR_14',
        #             '$ATR_28',
        #             '$BOP',
        #             '$CCI_14',
        #             '$CCI_28',
        #             '$CMO_14',
        #             '$CMO_28',
        #             '$DAILY_AMOUNT_RATIO',
        #             '$EMA_10',
        #             '$EMA_20',
        #             '$EMA_5',
        #             '$KAMA_12',
        #             '$KAMA_24',
        #             '$KAMA_48',
        #             '$LINEARREG_SLOPE_14',
        #             '$LINEARREG_SLOPE_28',
        #             '$LINEARREG_SLOPE_5',
        #             '$MACD',
        #             '$MACD_HIST',
        #             '$MACD_SIGNAL',
        #             '$MFI_12',
        #             '$MFI_24',
        #             '$MFI_48',
        #             '$MFI_6',
        #             '$MOM_12',
        #             '$MOM_24',
        #             '$MOM_48',
        #             '$MOM_6',
        #             '$NATR_14',
        #             '$NATR_28',
        #             '$OBV',
        #             '$RANK',
        #             '$ROC_12',
        #             '$ROC_24',
        #             '$ROC_48',
        #             '$ROC_6',
        #             '$RSI_12',
        #             '$RSI_24',
        #             '$RSI_6',
        #             '$SAR',
        #             '$SIZE',
        #             '$STOCHF_d',
        #             '$STOCHF_k',
        #             '$STOCHRSI_d',
        #             '$STOCHRSI_k',
        #             '$STR_FACTOR',
        #             '$TEMA_12',
        #             '$TEMA_24',
        #             '$TEMA_48',
        #             '$TRANGE',
        #             '$TRIMA_12',
        #             '$TRIMA_24',
        #             '$TRIMA_48',
        #             '$TRIX_12',
        #             '$TRIX_24',
        #             '$TRIX_48',
        #             '$TSF_10',
        #             '$TSF_20',
        #             '$TSF_40',
        #             '$TSF_5',
        #             '$TURN_MAX_10',
        #             '$TURN_MAX_20',
        #             '$TURN_MAX_40',
        #             '$TURN_MAX_5',
        #             '$TURN_MIN_10',
        #             '$TURN_MIN_20',
        #             '$TURN_MIN_40',
        #             '$TURN_MIN_5',
        #             '$TURN_RATE_EMA_10',
        #             '$TURN_RATE_EMA_20',
        #             '$TURN_RATE_EMA_5',
        #             '$TURN_RATE_LN',
        #             '$TURN_ROC_10',
        #             '$TURN_ROC_20',
        #             '$TURN_ROC_40',
        #             '$TURN_ROC_5',
        #             '$TURN_RSI_10',
        #             '$TURN_RSI_20',
        #             '$TURN_RSI_40',
        #             '$TURN_RSI_5',
        #             '$TURN_SLOPE_10',
        #             '$TURN_SLOPE_20',
        #             '$TURN_SLOPE_40',
        #             '$TURN_SLOPE_5',
        #             '$TURN_TSF_10',
        #             '$TURN_TSF_20',
        #             '$TURN_TSF_40',
        #             '$TURN_TSF_5',
        #             '$ULTOSC',
        #             '$VAR_10',
        #             '$VAR_20',
        #             '$VAR_40',
        #             '$VAR_5',
        #             '$WILLR_12',
        #             '$WILLR_24',
        #             '$WILLR_48',
        #             '$WILLR_6']
        
        # names += ['AD',
        #             'ADOSC',
        #             'ADX_14',
        #             'ADX_28',
        #             'AMOUNT_LN',
        #             'AMT_EMA_10',
        #             'AMT_EMA_20',
        #             'AMT_EMA_40',
        #             'AMT_EMA_5',
        #             'AMT_MAX_10',
        #             'AMT_MAX_20',
        #             'AMT_MAX_40',
        #             'AMT_MAX_5',
        #             'AMT_MIN_10',
        #             'AMT_MIN_20',
        #             'AMT_MIN_40',
        #             'AMT_MIN_5',
        #             'AMT_ROC_10',
        #             'AMT_ROC_20',
        #             'AMT_ROC_40',
        #             'AMT_ROC_5',
        #             'AMT_RSI_10',
        #             'AMT_RSI_20',
        #             'AMT_RSI_40',
        #             'AMT_RSI_5',
        #             'AMT_SLOPE_10',
        #             'AMT_SLOPE_20',
        #             'AMT_SLOPE_40',
        #             'AMT_SLOPE_5',
        #             'AMT_TRIX_10',
        #             'AMT_TRIX_20',
        #             'AMT_TRIX_40',
        #             'AMT_TRIX_5',
        #             'AMT_TSF_10',
        #             'AMT_TSF_20',
        #             'AMT_TSF_40',
        #             'AMT_TSF_5',
        #             'AMT_VAR_10',
        #             'AMT_VAR_20',
        #             'AMT_VAR_40',
        #             'AMT_VAR_5',
        #             'APO',
        #             'AROON_14_down',
        #             'AROON_14_up',
        #             'AROON_28_down',
        #             'AROON_28_up',
        #             'ATR_14',
        #             'ATR_28',
        #             'BOP',
        #             'CCI_14',
        #             'CCI_28',
        #             'CMO_14',
        #             'CMO_28',
        #             'DAILY_AMOUNT_RATIO',
        #             'EMA_10',
        #             'EMA_20',
        #             'EMA_5',
        #             'KAMA_12',
        #             'KAMA_24',
        #             'KAMA_48',
        #             'LINEARREG_SLOPE_14',
        #             'LINEARREG_SLOPE_28',
        #             'LINEARREG_SLOPE_5',
        #             'MACD',
        #             'MACD_HIST',
        #             'MACD_SIGNAL',
        #             'MFI_12',
        #             'MFI_24',
        #             'MFI_48',
        #             'MFI_6',
        #             'MOM_12',
        #             'MOM_24',
        #             'MOM_48',
        #             'MOM_6',
        #             'NATR_14',
        #             'NATR_28',
        #             'OBV',
        #             'RANK',
        #             'ROC_12',
        #             'ROC_24',
        #             'ROC_48',
        #             'ROC_6',
        #             'RSI_12',
        #             'RSI_24',
        #             'RSI_6',
        #             'SAR',
        #             'SIZE',
        #             'STOCHF_d',
        #             'STOCHF_k',
        #             'STOCHRSI_d',
        #             'STOCHRSI_k',
        #             'STR_FACTOR',
        #             'TEMA_12',
        #             'TEMA_24',
        #             'TEMA_48',
        #             'TRANGE',
        #             'TRIMA_12',
        #             'TRIMA_24',
        #             'TRIMA_48',
        #             'TRIX_12',
        #             'TRIX_24',
        #             'TRIX_48',
        #             'TSF_10',
        #             'TSF_20',
        #             'TSF_40',
        #             'TSF_5',
        #             'TURN_MAX_10',
        #             'TURN_MAX_20',
        #             'TURN_MAX_40',
        #             'TURN_MAX_5',
        #             'TURN_MIN_10',
        #             'TURN_MIN_20',
        #             'TURN_MIN_40',
        #             'TURN_MIN_5',
        #             'TURN_RATE_EMA_10',
        #             'TURN_RATE_EMA_20',
        #             'TURN_RATE_EMA_5',
        #             'TURN_RATE_LN',
        #             'TURN_ROC_10',
        #             'TURN_ROC_20',
        #             'TURN_ROC_40',
        #             'TURN_ROC_5',
        #             'TURN_RSI_10',
        #             'TURN_RSI_20',
        #             'TURN_RSI_40',
        #             'TURN_RSI_5',
        #             'TURN_SLOPE_10',
        #             'TURN_SLOPE_20',
        #             'TURN_SLOPE_40',
        #             'TURN_SLOPE_5',
        #             'TURN_TSF_10',
        #             'TURN_TSF_20',
        #             'TURN_TSF_40',
        #             'TURN_TSF_5',
        #             'ULTOSC',
        #             'VAR_10',
        #             'VAR_20',
        #             'VAR_40',
        #             'VAR_5',
        #             'WILLR_12',
        #             'WILLR_24',
        #             'WILLR_48',
        #             'WILLR_6']

        # ## v4.0
        # fields += ['$EMA_5',
        #            '$EMA_10',
        #            '$EMA_20',
        #            '$SAR',
        #            '$KAMA_12',
        #            '$KAMA_24',
        #            '$KAMA_48',
        #            '$TEMA_12',
        #            '$TEMA_24',
        #            '$TEMA_48',
        #            '$TRIMA_12',
        #            '$TRIMA_24',
        #            '$TRIMA_48',
        #            '$ADX_14',
        #            '$ADX_28',
        #            '$APO',
        #            '$AROON_14_down',
        #            '$AROON_14_up',
        #            '$AROON_28_down',
        #            '$AROON_28_up',
        #            '$BOP',
        #            '$CCI_14',
        #            '$CCI_28',
        #            '$CMO_14',
        #            '$CMO_28',
        #            '$MACD',
        #            '$MACD_SIGNAL',
        #            '$MACD_HIST',
        #            '$MOM_6',
        #            '$MOM_12',
        #            '$MOM_24',
        #            '$MOM_48',
        #            '$MFI_6',
        #            '$MFI_12',
        #            '$MFI_24',
        #            '$MFI_48',
        #            '$ROC_6',
        #            '$ROC_12',
        #            '$ROC_24',
        #            '$ROC_48',
        #            '$RSI_6',
        #            '$RSI_12',
        #            '$RSI_24',
        #            '$STOCHF_k',
        #            '$STOCHF_d',
        #            '$STOCHRSI_k',
        #            '$STOCHRSI_d',
        #            '$TRIX_12',
        #            '$TRIX_24',
        #            '$TRIX_48',
        #            '$ULTOSC',
        #            '$WILLR_6',
        #            '$WILLR_12',
        #            '$WILLR_24',
        #            '$WILLR_48',
        #            '$AD',
        #            '$ADOSC',
        #            '$OBV',
        #            '$ATR_14',
        #            '$ATR_28',
        #            '$NATR_14',
        #            '$NATR_28',
        #            '$TRANGE',
        #            '$TURN_RATE_5',
        #            '$TURN_RATE_10',
        #            '$TURN_RATE_20',
        #            '$TURN_RATE_MIX',
        #            '$TURN_ROC_6',
        #            '$TURN_ROC_12',
        #            '$TURN_ROC_24',
        #            '$TURN_ROC_48',
        #            '$BETA',
        #            '$DAILY_AMOUNT_RATIO',
        #            '$STR_FACTOR']
        
        # fields += ['$EMA_5',
        #            '$EMA_10', 
        #            '$EMA_20', 
        #            '$SAR', 
        #            '$RSI_6', 
        #            '$RSI_12', 
        #            '$RSI_24', 
        #            '$ADX_14', 
        #            '$ADX_28', 
        #            '$BOP', 
        #            '$CCI_14', 
        #            '$CCI_28', 
        #            '$MACD',
        #            '$MACD_SIGNAL',
        #            '$MACD_HIST',
        #            '$MOM_6', 
        #            '$MOM_12', 
        #            '$MOM_24', 
        #            '$MOM_48', 
        #            '$ULTOSC', 
        #            '$WILLR_6', 
        #            '$WILLR_12', 
        #            '$WILLR_24', 
        #            '$WILLR_48', 
        #            '$AD', 
        #            '$ADOSC', 
        #            '$OBV', 
        #            '$ATR_14', 
        #            '$ATR_28', 
        #            '$NATR_14', 
        #            '$NATR_28', 
        #            '$TRANGE']

        ## v4.0
        # names += ['EMA_5',
        #            'EMA_10',
        #            'EMA_20',
        #            'SAR',
        #            'KAMA_12',
        #            'KAMA_24',
        #            'KAMA_48',
        #            'TEMA_12',
        #            'TEMA_24',
        #            'TEMA_48',
        #            'TRIMA_12',
        #            'TRIMA_24',
        #            'TRIMA_48',
        #            'ADX_14',
        #            'ADX_28',
        #            'APO',
        #            'AROON_14_down',
        #            'AROON_14_up',
        #            'AROON_28_down',
        #            'AROON_28_up',
        #            'BOP',
        #            'CCI_14',
        #            'CCI_28',
        #            'CMO_14',
        #            'CMO_28',
        #            'MACD',
        #            'MACD_SIGNAL',
        #            'MACD_HIST',
        #            'MOM_6',
        #            'MOM_12',
        #            'MOM_24',
        #            'MOM_48',
        #            'MFI_6',
        #            'MFI_12',
        #            'MFI_24',
        #            'MFI_48',
        #            'ROC_6',
        #            'ROC_12',
        #            'ROC_24',
        #            'ROC_48',
        #            'RSI_6',
        #            'RSI_12',
        #            'RSI_24',
        #            'STOCHF_k',
        #            'STOCHF_d',
        #            'STOCHRSI_k',
        #            'STOCHRSI_d',
        #            'TRIX_12',
        #            'TRIX_24',
        #            'TRIX_48',
        #            'ULTOSC',
        #            'WILLR_6',
        #            'WILLR_12',
        #            'WILLR_24',
        #            'WILLR_48',
        #            'AD',
        #            'ADOSC',
        #            'OBV',
        #            'ATR_14',
        #            'ATR_28',
        #            'NATR_14',
        #            'NATR_28',
        #            'TRANGE',
        #            'TURN_RATE_5',
        #            'TURN_RATE_10',
        #            'TURN_RATE_20',
        #            'TURN_RATE_MIX',
        #            'TURN_ROC_6',
        #            'TURN_ROC_12',
        #            'TURN_ROC_24',
        #            'TURN_ROC_48',
        #            'BETA',
        #            'DAILY_AMOUNT_RATIO',
        #            'STR_FACTOR']

        # names += ['EMA5', 
        #           'EMA10', 
        #           'EMA20', 
        #           'SAR', 
        #           'RSI6', 
        #           'RSI12', 
        #           'RSI24', 
        #           'ADX14', 
        #           'ADX28', 
        #           'BOP', 
        #           'CCI14', 
        #           'CCI28', 
        #           'MACD',
        #           'MACD_SIGNAL',
        #           'MACD_HIST',
        #           'MOM6', 
        #           'MOM12', 
        #           'MOM24', 
        #           'MOM48', 
        #           'ULTOSC', 
        #           'WILLR6', 
        #           'WILLR12', 
        #           'WILLR24', 
        #           'WILLR48', 
        #           'AD', 
        #           'ADOSC', 
        #           'OBV', 
        #           'ATR14', 
        #           'ATR28', 
        #           'NATR14', 
        #           'NATR28', 
        #           'TRANGE']    
        return fields, names

    def get_label_config(self):
        return ["Ref($open, -2)/Ref($open, -1) - 1"], ["LABEL0"]
    
class MyAlpha158_DyFeature(DataHandlerLP):
    """通过动态传入feature meta文件，实现动态feature的DataHandlerLP

    Args:
        DataHandlerLP (_type_): _description_
    """
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        feature_meta_file = None,
        **kwargs
    ):
        if feature_meta_file is None:
            raise ValueError("feature meta file should be provided")
        self.dynamic_feature_dic = self._load_feature_meta(feature_meta_file)

        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }     
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs
        )

    def get_feature_count(self):
        return len(self.get_feature_config()[0])

    def _load_feature_meta(self, meta_file):
        with open(meta_file, 'r') as f:
            meta = json.load(f)
        ##check json file content
        if 'fields' not in meta or 'names' not in meta:
            print(f"{meta_file} should contain 'fields' and 'names' keys")
            print("meta file path: {meta_file}")
            print("meta file content:\n {meta}")
            raise ValueError("feature meta file should contain 'fields' and 'names' keys")
        
        ## check fields data type
        if not isinstance(meta['fields'], list):
            print(f"{meta_file} fields should be a list")
            print("meta file path: {meta_file}")
            print("meta file content:\n {meta}")
            raise ValueError("feature meta file fields should be a list")
        
        ## check names data type
        if not isinstance(meta['names'], list):
            print(f"{meta_file} names should be a list")
            print("meta file path: {meta_file}")
            print("meta file content:\n {meta}")
            raise ValueError("feature meta file names should be a list")

        ## check fields data format
        if meta['fields'][0].startswith('$') is False:
            print(f"{meta_file} fields format should start with '$'")
            print("meta file path: {meta_file}")
            print("meta file content:\n {meta}")
            raise ValueError("feature meta file fields format should start with '$'")

        ## fields and names should have the same length
        if meta['fields'] and meta['names'] and len(meta['fields']) != len(meta['names']):
            print(f"{meta_file} fields and names should have the same length")
            print("meta file path: {meta_file}")
            print("meta file content:\n {meta}")
            raise ValueError("feature meta file fields and names should have the same length")

        return meta

    def get_feature_config(self):
        fields, names = [],[]
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {},
        }
        fields, names = Alpha158DL.get_feature_config(conf)
        
        fields += ['$turn','$peTTM', '$pbMRQ', '$psTTM']
        fields += self.dynamic_feature_dic['fields']
        
        names += ['TURNOVER', 'PETTM', 'PBMQR', 'PSTTM']
        names += self.dynamic_feature_dic['names']
        
        return fields, names

    def get_label_config(self):
        return ["Ref($open, -2)/Ref($open, -1) - 1"], ["LABEL0"]

class MyAlphaV4(DataHandlerLP):
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        **kwargs
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs
        )
    def get_feature_config(self):
        fields, names = [],[]
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {},
        }
        fields, names = Alpha158DL.get_feature_config(conf)
        
        fields += ['$turn','$peTTM', '$pbMRQ', '$psTTM']
        names += ['TURNOVER', 'PETTM', 'PBMQR', 'PSTTM']
        ## add baostock basic features
        ## turn / peTTM / pbMRQ / psTTM / pcfNcfTTM / pbMRQ
        
        # fields += ['Ref($turn, 1)/$turn', '$turn', '$peTTM', '$pbMRQ', '$psTTM', '$pcfNcfTTM', '$pbMRQ']
        # names += ['TURNOVER1', 'TURNOVER0', 'PETTM', 'PBMQR', 'PSTTM', 'PCFNCF', 'PBMQR']
        
        # ## v4.0
        fields += [
            '$EMA_5', '$EMA_10', '$EMA_20', '$SAR', '$KAMA_12',
            '$KAMA_24', '$KAMA_48', '$TEMA_12', '$TEMA_24', '$TEMA_48',
            '$TRIMA_12', '$TRIMA_24', '$TRIMA_48', '$ADX_14', '$ADX_28',
            '$APO', '$AROON_14_down', '$AROON_14_up', '$AROON_28_down', '$AROON_28_up',
            '$BOP', '$CCI_14', '$CCI_28', '$CMO_14', '$CMO_28',
            '$MACD', '$MACD_SIGNAL', '$MACD_HIST', '$MOM_6', '$MOM_12',
            '$MOM_24', '$MOM_48', '$MFI_6', '$MFI_12', '$MFI_24',
            '$MFI_48', '$ROC_6', '$ROC_12', '$ROC_24', '$ROC_48',
            '$RSI_6', '$RSI_12', '$RSI_24', '$STOCHF_k', '$STOCHF_d',
            '$STOCHRSI_k', '$STOCHRSI_d', '$TRIX_12', '$TRIX_24', '$TRIX_48',
            '$ULTOSC', '$WILLR_6', '$WILLR_12', '$WILLR_24', '$WILLR_48',
            '$AD', '$ADOSC', '$OBV', '$ATR_14', '$ATR_28',
            '$NATR_14', '$NATR_28', '$TRANGE', '$DAILY_AMOUNT_RATIO', '$STR_FACTOR',
            '$norm_RSRS', '$pos_RSRS' ]

        # v4.0
        names += [
            'EMA_5', 'EMA_10', 'EMA_20', 'SAR', 'KAMA_12',
            'KAMA_24', 'KAMA_48', 'TEMA_12', 'TEMA_24', 'TEMA_48',
            'TRIMA_12', 'TRIMA_24', 'TRIMA_48', 'ADX_14', 'ADX_28',
            'APO', 'AROON_14_down', 'AROON_14_up', 'AROON_28_down', 'AROON_28_up',
            'BOP', 'CCI_14', 'CCI_28', 'CMO_14', 'CMO_28',
            'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'MOM_6', 'MOM_12',
            'MOM_24', 'MOM_48', 'MFI_6', 'MFI_12', 'MFI_24',
            'MFI_48', 'ROC_6', 'ROC_12', 'ROC_24', 'ROC_48',
            'RSI_6', 'RSI_12', 'RSI_24', 'STOCHF_k', 'STOCHF_d',
            'STOCHRSI_k', 'STOCHRSI_d', 'TRIX_12', 'TRIX_24', 'TRIX_48',
            'ULTOSC', 'WILLR_6', 'WILLR_12', 'WILLR_24', 'WILLR_48',
            'AD', 'ADOSC', 'OBV', 'ATR_14', 'ATR_28',
            'NATR_14', 'NATR_28', 'TRANGE', 'DAILY_AMOUNT_RATIO', 'STR_FACTOR',
            'norm_RSRS', 'pos_RSRS']

        return fields, names

    def get_label_config(self):
        return ["Ref($open, -2)/Ref($open, -1) - 1"], ["LABEL0"]
    
    def get_feature_count(self):
        return len(self.get_feature_config()[0])
    