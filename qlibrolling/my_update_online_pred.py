# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This example shows how OnlineTool works when we need update prediction.
There are two parts including first_train and update_online_pred.
Firstly, we will finish the training and set the trained models to the `online` models.
Next, we will finish updating online predictions.
"""
import copy
import fire
import qlib
from qlib.constant import REG_CN
from qlib.model.trainer import task_train
from qlib.workflow.online.utils import OnlineToolR
from qlib.tests.config import CSI300_GBDT_TASK
from qlib.data import D
import datetime
import pandas as pd

# task = copy.deepcopy(CSI300_GBDT_TASK)

class UpdatePredExample:
    def __init__(
        self,
        provider_uri=None,
        region=REG_CN,
        experiment_name="online_srv",
        stock_pool="csi300",
        horizon=20,
    ):
        qlib.init(provider_uri=provider_uri, region=region)
        self.stock_pool = stock_pool  # 股池
        # 定义任务
        self.horizon = horizon  # 预测期长度
        cal = D.calendar(freq="day")
        # print(cal)
        latest_date = cal[-1]  # 日历中最近的一天

        valid_length = 252 * 2  # 验证集长度多少天
        test = (
            latest_date,
            latest_date,
        )  # 日历中最后一天，该天结束后，用户要更新qlib数据，然后根据该日信息预测该日的未来收益率
        valid_end = cal[-(self.horizon + 1)]  # 验证集开始时间，与测试集要保持必要的间隔，防止信息泄露
        valid_start = cal[-(self.horizon + 1) - valid_length]  # valid_end前2年
        train_start = pd.to_datetime("2008-01-01")  # 训练集最好从股权分置改革后开始，即2008-01-01.
        train_end = cal[-(self.horizon + 1) - valid_length - 1]  # 为验证集开始时间前一天
   
        model_class = "LinearModel"
        if model_class == "LinearModel": 
            # 线性模型没有early stoping机制，不需要验证集，故合并到训练集
            train = (train_start, valid_end) 
        else:
            # 对有early stoping机制的模型，需要验证集
            train = (train_start, train_end)
        
        valid = (valid_start, valid_end)
        print("/n========================")
        print("train", train)
        print("valid", valid)
        print("test", test)
        print("========================")

        task = {}
        task["model"] = {
                "class": model_class,
                "module_path": "qlib.contrib.model.linear",
                "kwargs": {
                    "estimator": "ridge",
                    "alpha": 0.05,
                },
            }
        
        task["dataset"] = {
                "class": "DatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": {
                        "class": "Alpha158",
                        "module_path": "qlib.contrib.data.handler",
                        "kwargs": {
                            "start_time": train[0],
                            "end_time": test[1],
                            "fit_start_time": train[0] ,                            
                            "fit_end_time": train[1]  ,
                            "instruments": self.stock_pool,
                            "infer_processors": [
                                {
                                    "class": "RobustZScoreNorm",
                                    "kwargs": {
                                        "fields_group": "feature",
                                        "clip_outlier": True,
                                    },
                                },
                                {
                                    "class": "Fillna",
                                    "kwargs": {"fields_group": "feature"},
                                },
                            ],
                            "learn_processors": [
                                {"class": "DropnaLabel"},
                                {
                                    "class": "CSRankNorm",
                                    "kwargs": {"fields_group": "label"},
                                },
                            ],
                            "label": [
                                "Ref($open, -{}) / Ref($open, -1) - 1".format(
                                    horizon + 1
                                )
                            ],
                        },
                    },
                    "segments": {
                        "train": train,
                        "valid": valid,
                        "test": test,
                    },
                },
            }
        task["record"] =  {
                "class": "SignalRecord",
                "module_path": "qlib.workflow.record_temp",
            }


        self.experiment_name = experiment_name
        self.online_tool = OnlineToolR(self.experiment_name)
        self.task_config = task

    def first_train(self):
        rec = task_train(self.task_config, experiment_name=self.experiment_name)
        self.online_tool.reset_online_tag(rec)  # set to online model

    def update_online_pred(self):
        self.online_tool.update_online_pred()

    def main(self):
        self.first_train()
        self.update_online_pred()  # 默认从test[0]预测到数据集中最新的日期


if __name__ == "__main__":

    u = UpdatePredExample(
        provider_uri=r"G:\qlibrolling\qlib_data\cn_data_rolling",
        stock_pool="csi300",
        horizon=5,
    )
    u.main()

    rec = u.online_tool.online_models()[0]  # 得到recorder
    print("recorder====", rec)
    print("experiment_id",rec.experiment_id, "record id",rec.id)

    # 打印预测结果
    from qlib.workflow import R
    predict_recorder = R.get_recorder(recorder_id=rec.id, experiment_id=rec.experiment_id) 
    pred_df = predict_recorder.load_object('pred.pkl')
    print(pred_df.sort_values("score"))
    print("experiment_id",rec.experiment_id, "record id",rec.id)

