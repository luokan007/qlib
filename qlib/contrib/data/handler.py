# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from qlib.contrib.data.loader import Alpha158DL, Alpha360DL
from ...data.dataset.handler import DataHandlerLP
from ...data.dataset.processor import Processor
from ...utils import get_callable_kwargs
from ...data.dataset import processor as processor_module
from inspect import getfullargspec


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
        
        # ## add ta-lib features
        # # 'EMA': {'periods': [5, 10, 20]},
        # # 'SAR': {},

        # # 'RSI': {'periods': [6, 12, 24]},
        # # 'ADX': {'timeperiod': [14,28]},
        # # 'BOP': {},
        # # 'CCI': {'timeperiod': [14,28]},
        # # 'MACD': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9},
        # # 'MOM': {'timeperiod': [6, 12, 24, 48]},
        # # 'ULTOSC': {},
        # # 'WILLR': {'timeperiod': [6, 12, 24, 48]},

        # # 'AD': {},
        # # 'ADOSC': {'fastperiod': 3, 'slowperiod': 10},
        # # 'OBV': {},

        # # 'ATR': {'timeperiod': [14, 28]},
        # # 'NATR': {'timeperiod': [14, 28]},
        # # 'TRANGE': {}

        
        fields += ['$EMA_5', 
                   '$EMA_10', 
                   '$EMA_20', 
                   '$SAR', 
                   '$RSI_6', 
                   '$RSI_12', 
                   '$RSI_24', 
                   '$ADX_14', 
                   '$ADX_28', 
                   '$BOP', 
                   '$CCI_14', 
                   '$CCI_28', 
                   '$MACD',
                   '$MACD_SIGNAL',
                   '$MACD_HIST',
                   '$MOM_6', 
                   '$MOM_12', 
                   '$MOM_24', 
                   '$MOM_48', 
                   '$ULTOSC', 
                   '$WILLR_6', 
                   '$WILLR_12', 
                   '$WILLR_24', 
                   '$WILLR_48', 
                   '$AD', 
                   '$ADOSC', 
                   '$OBV', 
                   '$ATR_14', 
                   '$ATR_28', 
                   '$NATR_14', 
                   '$NATR_28', 
                   '$TRANGE']

        names += ['EMA5', 
                  'EMA10', 
                  'EMA20', 
                  'SAR', 
                  'RSI6', 
                  'RSI12', 
                  'RSI24', 
                  'ADX14', 
                  'ADX28', 
                  'BOP', 
                  'CCI14', 
                  'CCI28', 
                  'MACD',
                  'MACD_SIGNAL',
                  'MACD_HIST',
                  'MOM6', 
                  'MOM12', 
                  'MOM24', 
                  'MOM48', 
                  'ULTOSC', 
                  'WILLR6', 
                  'WILLR12', 
                  'WILLR24', 
                  'WILLR48', 
                  'AD', 
                  'ADOSC', 
                  'OBV', 
                  'ATR14', 
                  'ATR28', 
                  'NATR14', 
                  'NATR28', 
                  'TRANGE']    
        return fields, names

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]
