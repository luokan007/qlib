import pandas as pd
import copy
import datetime
import csv
from pathlib import Path
from typing import Union
from datetime import datetime

import qlib
from qlib.contrib.rolling.base import Rolling
from qlib.workflow import R
from qlib.data import D # 基础行情数据服务的对象
from qlib.constant import REG_CN
from qlib.model.trainer import task_train
from qlib.workflow.online.utils import OnlineToolR

import quantstats as qs 
# from qlib.tests.data import GetData
# import fire
# from qlib import auto_init

  
#######
# fixed issue: 中文环境会导致Monthly Return表格展示全部为零，具体可见下方链接
# 修复方法：将locale设置为en_US.UTF-8
# detailed information: https://github.com/ranaroussi/quantstats/issues/255
##
import os
import locale
# 设置语言环境为 en_US.UTF-8
os.environ['LC_ALL'] = 'en_US.UTF-8'
os.environ['LANG'] = 'en_US.UTF-8'
# 确保设置生效
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

#current_file_path = os.path.abspath(__file__)
#DIR_NAME = os.path.dirname(current_file_path)
DIR_NAME = "/home/godlike/project/GoldSparrow"


MODEL_CONFIG_FOLDER = DIR_NAME + "/Online_Model"
CONF_LIST = [MODEL_CONFIG_FOLDER + "/workflow_config_lstm_Alpha158.yaml",
             MODEL_CONFIG_FOLDER + "/workflow_config_linear_Alpha158.yaml",
             MODEL_CONFIG_FOLDER + "/workflow_config_lightgbm_Alpha158.yaml"]
REPORT_FILE_PATH = DIR_NAME + "/Offline_Report/report_lgbm.html"

class RollingBenchmark(Rolling):
    """_summary_

    Args:
        Rolling (_type_): _description_
    """

    def __init__(self,
                 conf_path: Union[str, Path] = None,
                 horizon=20,
                 **kwargs) -> None:

        print('conf_path', conf_path)
        super().__init__(conf_path=conf_path, horizon=horizon, **kwargs)

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


def Offline_Evaluate(is_process_data = True) -> None:
    """_summary_

    Args:
        is_process_data (bool, optional): _description_. Defaults to True.
        is_train (bool, optional): _description_. Defaults to True.
    """
    
    #####################################
    # 0 删除缓存数据集handler pkl文件
    #####################################
    import os
    from os import listdir
    pkl_path = os.path.dirname(__file__)  # 当前文件所在的目录
    for file_name in listdir(pkl_path):
        if file_name.endswith('.pkl'):
            os.remove(pkl_path + '/' + file_name)
            
    
    ###################################
    # 1 滚动训练与预测
    ###################################
    
    exp_name = "combine"  # 合并预测结果pred.pkl存放mlflow实验名
    rb = RollingBenchmark(
        conf_path=CONF_LIST[2],
        step=40,  # 滚动步长，每隔40天滚动训练一次，它也决定了每滚测试集长度为40天
        horizon=20,  # 收益率预测期长度
        exp_name=exp_name)  # 最终合并预测结果所在实验名

    config_dict = rb._raw_conf()  # 配置字典
    # 初始化qlib
    qlib.init(provider_uri=config_dict["qlib_init"]["provider_uri"],
              region=config_dict["qlib_init"]["region"])

    # 滚动训练与预测
    rb._train_rolling_tasks()

    #################################
    # 2 滚动预测结果合并成大预测结果：每步小测试期预测结果合并成大测试期预测结果
    #################################
    rb._ens_rolling()
    # 打印合并后预测结果文件所在实验id，实验名和记录id
    print('experiment_id', rb._experiment_id, 'exp_name', exp_name, 'rid',
          rb._rid)

    #################################
    # 3 qlib信号分析与回测：在大测试期里执行信号分析与回测
    #################################
    # 回测:记录信号分析结果（如IC等）和回测结果（如仓位情况等）
    rb._update_rolling_rec()

    # 打印合并后预测结果文件所在实验id，实验名和记录id。回测结果也在此实验和记录id下。
    print('experiment_id', rb._experiment_id, 'exp_name', exp_name, 'rid',
          rb._rid)
    
    ##########################
    # 4. 回测结果导出到quantstate
    
    predict_recorder = R.get_recorder(recorder_id = rb._rid, 
                                      experiment_name = exp_name)
    # 回测结果提取到df
    report_normal_1day_df = predict_recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    returns = report_normal_1day_df['return']
    
    # benchmark结果提取
    df_benchmark = D.features(['sh000300'], fields=['$close','$change','$factor'])
    df_benchmark_returns = df_benchmark.loc['sh000300']['$close'].pct_change()
    df_benchmark['origClose'] = df_benchmark['$close'] /  df_benchmark['$factor']
    
    #returns
    qs.reports.html(returns, benchmark = df_benchmark_returns.loc[returns.index], output=REPORT_FILE_PATH, rf=0.0)
    
    return None


class GenerateOrder:
    """_summary_
    """
    def __init__(
        self,
        predict_recorder,
        top_k = 50,
        n_drop = 5,
        working_dir = "/home/godlike/project/GoldSparrow/Online_Order/",
        position_csv ="position.csv",
        buy_order_csv = "buy.csv",
        sell_order_csv = "sell.csv"
        ):
        self.predict_recorder = predict_recorder
        self.working_dir = working_dir
        self.top_k = top_k
        self.n_drop = n_drop
        self.position_csv = position_csv
        self.buy_order_csv = buy_order_csv
        self.sell_order_csv = sell_order_csv
        self.forbid_list = []
        
        self.risk_degree = 0.95
        self.stamp_duty = 0.001 #印花税率
        self.commission = 0.0005 # 佣金率
        self.cash = 55448
         
    def generate_order_csv(self, predict_recorder=None):
        """_summary_

        Args:
            predict_recorder (_type_, optional): _description_. Defaults to None.
        """
        if predict_recorder is None:
            prc = self.predict_recorder
        else:
            prc = predict_recorder
            
        working_dir = self.working_dir
        
        pred_df = prc.load_object("pred.pkl")
        print("pred_df",pred_df)
        
        latest_date = pred_df.index.get_level_values('datetime').max()
        
        print("the latest date is:",latest_date)
        
        latest_date_str = latest_date.strftime("%Y_%m_%d")
        order_folder_name = os.path.join(working_dir, latest_date_str)
        if not os.path.exists(order_folder_name):
            os.mkdir(order_folder_name)
        
        pred_score = pred_df.xs(latest_date) # 最新一个交易日各股预测分，df
        pred_score = pred_score[
            ~pred_score.index.isin(self.forbid_list)] # 股池去掉禁止持有的股票
        
        ##获得当前持仓
        initial_position_csv_file_path = os.path.join(order_folder_name, self.position_csv)
        if os.path.exists(initial_position_csv_file_path):
            initial_position_df = pd.read_csv(initial_position_csv_file_path, index_col="code")
            initial_position_df.index = initial_position_df.index.str.upper()
            current_stock_list = list(initial_position_df.index)
        else:
            current_stock_list = []
            print("no position file, start with empty position")
            initial_position_df = pd.DataFrame(columns=["code", "amount"])
        
        ##获得当前累积资产
        total_asset = self.cash
        for index, row in initial_position_df.iterrows():
            total_asset += row['quantity'] * row['close']
        print("total asset: ",total_asset)
            
        # 今日已持仓股票列表，按score降序排列。若某持仓股票不在pred_score中，则该股票排在index最后。index类型
        last_tuple = (
            pred_score.reindex(current_stock_list).
            sort_values(by="score", ascending=False, na_position="last").
            index)
        
        # 股池pred_score中，去掉已持仓的股票列表，index类型，按score降序
        new_tuple = (
            pred_score[~pred_score.index.isin(last_tuple)]
            .sort_values(by="score", ascending=False)
            .index)
        
        # 取new 的头 topk - (len(last) - n_drop)个股票，index类型，按score降序。这个数量是现有持仓last中，去掉最大卖出数量n_drop，要补齐到topk，需要买入的量。
        min_left = len(last_tuple) - self.n_drop  # n_drop是最大可能卖出支数，卖出n_drop后最小剩余持仓支数minLeft
        max_buy = self.top_k - min_left  # 最大可能买入支数，使得最终持仓达到topk
        today = new_tuple[:max_buy]
        
        # last和today的并集，index类型，按score降序
        comb = (
            pred_score.reindex(last_tuple.union(today))
            .sort_values(by="score", ascending=False, na_position="last")
            .index)

        # comb中后n_drop个股票，需要卖出。index类型
        sell = last_tuple[last_tuple.isin(comb[-self.n_drop:])]
        # today中头 topk - (len(last) -len(sell))个股票. 买入数量为现有持仓last中，去掉卖出数量len(sell)，要补齐到topk，需要买入的量。index类型
        left = len(last_tuple) - len(sell)  # 卖出len(sell)支股票后的剩余持仓支数
        need_buy = self.top_k - left  # 持仓提升到topk实际需要买入的支数
        buy = today[:need_buy]
        
        sell_order = []
        buy_order = []

        sell_order_file_path = os.path.join(order_folder_name, self.sell_order_csv)
        with open(sell_order_file_path, "w", newline="") as sell_csv_file:
            writer = csv.writer(sell_csv_file)
            # 先写入columns_name
            writer.writerow(["code", "quantity", "close"])
            
            for item in sell:
                # 卖的数量
                sell_amount = initial_position_df.loc[item]["quantity"]  # self.getposition(item).size
                sell_order.append((item, sell_amount))
                close = initial_position_df.loc[item]["close"]
                writer.writerow([item, sell_amount, close])
                trade_value = close * sell_amount  # 用今日收盘价估算明日开盘可能的成交金额

                trade_cost = trade_value * (2 * self.commission + self.stamp_duty)  # 估计交易成本
                self.cash += trade_value - trade_cost  # 估计现金累积值

        # 为要买入的股票每支分配的资金
        to_be_used_cash = self.cash - total_asset*( 1 - self.risk_degree)
        cash_per_stock = round(to_be_used_cash / len(buy) if len(buy) > 0 else 0, 2)
        
        # 买入操作
        buy_order_file_path = os.path.join(order_folder_name, self.buy_order_csv)
        with open(buy_order_file_path, "w", newline="") as buy_csv_file:
            writer = csv.writer(buy_csv_file)
            # 先写入columns_name
            writer.writerow(["code", "value"])
            for item in buy:
                writer.writerow([item, cash_per_stock])
                buy_order.append((item, cash_per_stock))
                
        # 更新新的持仓
        updated_position_df = initial_position_df.drop(index=sell)
        for stock_code, amount in buy_order:
            assert stock_code not in updated_position_df.index
            updated_position_df.loc[stock_code] = {'quantity': -1, 'close': -1}  ## 目前数据结构没有价格信息，待后续完善
        
        # 输出新的持仓文件
        new_position_csv_file_path = os.path.join(order_folder_name, f"new_{self.position_csv}")
        updated_position_df.to_csv(new_position_csv_file_path)
        print(f"New position file saved to {new_position_csv_file_path}")

if __name__ == "__main__":
    
    #Offline_Evaluate()  ###滚动模型的离线评测
    
    u = UpdatePredExample(
        provider_uri=r"/home/godlike/project/GoldSparrow/Updated_Stock_Data",
        stock_pool="csi300",
        horizon=5,
    )
    
    #u.first_train()
    u.update_online_pred()
    
    rec = u.online_tool.online_models()[0]  # 得到recorder
    print("recorder====", rec)
    print("experiment_id",rec.experiment_id, "record id",rec.id)

    # 打印预测结果
    #predict_recorder = R.get_recorder(recorder_id=rec.id, experiment_id=rec.experiment_id) 
    #pred_df = predict_recorder.load_object('pred.pkl')
    #print(pred_df.sort_values("score"))
    #print("experiment_id",rec.experiment_id, "record id",rec.id)
    
    g = GenerateOrder(predict_recorder = rec)
    g.generate_order_csv()
    
# cd /home/godlike/project/GoldSparrow    
# wget https://github.com/chenditc/investment_data/releases/download/2024-11-25/qlib_bin.tar.gz
# 
# tar -zxvf qlib_bin.tar.gz --strip-components=1
# 
    