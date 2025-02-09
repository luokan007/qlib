# title: back_test.py
# updated: 2024.12.28
# change log:
#   - 构建基于backtrader的回测框架
#   - 拆分topkdropout策略，新创建BackTest类
#   - 从实验ID、预测文件、预测DataFrame创建回测实例，提升未来代码的复用性
#   - 优化代码结构

# 目标：
#   1. 载入qlib数据，载入日频数据
#   2. 支持backtrader框架
#   3. 以开盘价买入
#   4. 增加滑点、
#   5. 改进risk_degree的计算方式
import os
import json
from datetime import datetime
import pandas as pd

import quantstats as qs
import backtrader as bt
# 用于扩展DataFeed
import qlib
from qlib.workflow import R
from qlib.data import D  # 基础行情数据服务的对象

from bt_topkdropout_strategy import TopkDropoutStrategy
from bt_topkdropout_strategy import StampDutyCommissionScheme
from contextlib import suppress


class BackTest:
    """回测类，支持多种初始化方式和参数配置"""
    
    def __init__(self, 
                 initial_cash=10000000.0,
                 bench_symbol='SH000300',
                 provider_uri=None,
                 report_name="quantstats-tearsheet.html",
                 slippage=0.0001):
        """
        Args:
            initial_cash (float): 初始资金
            bench_symbol (str): 基准标的代码
            provider_uri (str): 数据源路径
            report_name (str): 回测报告文件名
            slippage (float): 滑点率
        """
        self.cerebro = bt.Cerebro(stdstats=False)
        self.predictions = None
        self.start_date = None
        self.end_date = None
        self.initial_cash = initial_cash
        self.bench_symbol = bench_symbol
        self.provider_uri = provider_uri
        self.report_name = report_name
        self.slippage = slippage

    @classmethod
    def from_experiment(cls, provider_uri, exp_name, rid, **kwargs):
        """从实验ID创建回测实例
        
        Args:
            provider_uri (str): 数据源路径
            exp_name (str): 实验名称
            rid (str): 实验记录ID
            **kwargs: 其他参数，传递给__init__
        """
        instance = cls(provider_uri=provider_uri, **kwargs)
        qlib.init(provider_uri=provider_uri, region="cn")
        
        predict_recorder = R.get_recorder(recorder_id=rid, experiment_name=exp_name)
        pred_df = predict_recorder.load_object('pred.pkl')
        
        instance.predictions = pred_df
        instance.start_date = pred_df.index[0][0]
        instance.end_date = pred_df.index[-1][0]
        return instance

    @classmethod
    def from_pred_file(cls, provider_uri, pred_file_path, **kwargs):
        """从预测文件创建回测实例
        
        Args:
            provider_uri (str): 数据源路径
            pred_file_path (str): 预测文件路径
            **kwargs: 其他参数，传递给__init__
        """
        instance = cls(provider_uri=provider_uri, **kwargs)
        qlib.init(provider_uri=provider_uri, region="cn")
        
        pred_df = pd.read_pickle(pred_file_path)
        instance.predictions = pred_df
        instance.start_date = pred_df.index[0][0]
        instance.end_date = pred_df.index[-1][0]
        return instance

    @classmethod
    def from_predictions(cls, provider_uri, predictions_df, **kwargs):
        """从预测DataFrame创建回测实例
        
        Args:
            provider_uri (str): 数据源路径
            predictions_df (pd.DataFrame): 预测数据
            **kwargs: 其他参数，传递给__init__
        """
        instance = cls(provider_uri=provider_uri, **kwargs)
        qlib.init(provider_uri=provider_uri, region="cn")
        
        instance.predictions = predictions_df
        instance.start_date = predictions_df.index[0][0]
        instance.end_date = predictions_df.index[-1][0]
        return instance

    def initialize_backtest(self):
        """初始化回测环境和数据"""
        # 1. 准备数据
        stock_pool = list(self.predictions.index.levels[1])
        df_all = D.features(
            instruments=stock_pool,
            fields=['$open', '$high', '$low', '$close', '$change', '$factor','$volume'],
            start_time=self.start_date,
            end_time=self.end_date,
        )
        df_all = df_all.dropna(subset=["$open", "$high", "$low", "$close"])
        ##将ohlc价格修改为除权前的价格
        df_all['$open'] = df_all['$open']/df_all['$factor']
        df_all['$high'] = df_all['$high']/df_all['$factor']
        df_all['$low'] = df_all['$low']/df_all['$factor']
        df_all['$close'] = df_all['$close']/df_all['$factor']

        missing_stocks = set(stock_pool) - set(df_all.index.get_level_values(0).unique())
        if missing_stocks:
            print ("Missing stock count:", len(missing_stocks))
            print(f"Warning: The following stocks are missing from df_all: {missing_stocks}")

        
        # 2. 添加数据到回测引擎
        for stock in stock_pool:
            with suppress(KeyError):
                df_stock = df_all.xs(stock, level=0)
                data_feed = bt.feeds.PandasDirectData(
                    dataname=df_stock,
                    datetime=0,
                    open=1,
                    high=2,
                    low=3,
                    close=4,
                    volume=7,
                    openinterest=-1,
                    fromdate=self.start_date,
                    todate=self.end_date,
                    plot=False)
                self.cerebro.adddata(data_feed, name=stock)

        # 3. 设置回测参数
        self.cerebro.addstrategy(TopkDropoutStrategy, pred_df_all=self.predictions)
        self.cerebro.broker.set_checksubmit(False)
        self.cerebro.broker.setcash(10000000.0)
        comminfo = StampDutyCommissionScheme()
        self.cerebro.broker.addcommissioninfo(comminfo)
        self.cerebro.addanalyzer(bt.analyzers.PyFolio, _name='PyFolio') # 加入PyFolio分析者
        self.cerebro.broker.set_slippage_perc(0.0001)  # 百分比滑点


    def run(self):
        """运行回测"""
        starttime = datetime.now()

        self.initialize_backtest()  # 使用新的初始化方法

        results = self.cerebro.run()
        print('最终市值: %.2f' % self.cerebro.broker.getvalue())
        # PyFolio 分析
        strat = results[0]
        portfolio_stats = strat.analyzers.getbyname('PyFolio')
        returns, positions, transactions, gross_lev = portfolio_stats.get_pf_items()
        returns.index = returns.index.tz_convert(None)
        # 获取基准数据
        df_bench = D.features(
            [self.bench_symbol],
            fields=['$close'],
            start_time=self.start_date,
            end_time=self.end_date,
        ).xs(self.bench_symbol).pct_change().rename(columns={'$close': self.bench_symbol})

        # 生成分析报告
        qs.reports.html(returns, benchmark=df_bench, output=self.report_name)

        print('耗时', datetime.now()-starttime)

        return

        # return {
        #     'returns': returns,
        #     'benchmark': df_bench,
        #     'positions': positions,
        #     'transactions': transactions
        # }

def main():
    # # 使用示例
    # backtest = BackTest.from_experiment(
    #     provider_uri="/home/godlike/project/GoldSparrow/Updated_Stock_Data",
    #     exp_name="LSTM_CSI300_Alpha58",
    #     rid="0833139cd23a48d592f1a1c6510f8495"
    # )
    # backtest.run()
    
    ## 基于自定义预测文件的回测
    # report_file_path = '/home/godlike/project/GoldSparrow/Offline_Report/self_alpha_LSTM_v2.html'
    # output_dir = "/home/godlike/project/GoldSparrow/Temp_Data"
    #provider_uri = "/home/godlike/project/GoldSparrow/Day_Data/Day_data/qlib_data"
     # 配置文件路径
     
    # work_dir = '/root/autodl-tmp/GoldSparrow/Temp_Data'
    # provider_uri = "/root/autodl-tmp/GoldSparrow/Day_data/qlib_data"
    
    work_dir = '/home/godlike/project/GoldSparrow/Temp_Data'
    provider_uri = "/home/godlike/project/GoldSparrow/Day_Data/Day_data/qlib_data"

    config_file_name = 'config_20250209134215.json'#'config_20250209121635.json'
    report_file_name = 'report_20250209134215.html'#'report_20250209121635.html'
    
    config_file_path = os.path.join(work_dir, config_file_name)    
    report_file_path = os.path.join(work_dir, report_file_name)
    
    # 读取配置文件
    with open(config_file_path, 'r') as f:
        config = json.load(f)
    
    # 从配置文件中获取 pkl_path
    pkl_path = config['prediction_pkl']
    backtest = BackTest.from_pred_file(provider_uri=provider_uri, pred_file_path=pkl_path, report_name=report_file_path)
    backtest.run()
    
    # ### 基于1.3版本产生的experiment id的回测
    # report_file_path = '/home/godlike/project/GoldSparrow/Offline_Report/LSTM_benchmark.html'
    # provider_uri = "/home/godlike/project/GoldSparrow/Updated_Stock_Data"
    # backtest = BackTest.from_experiment(provider_uri = provider_uri, 
    #                                     exp_name="LSTM_CSI300_Alpha58",
    #                                     rid = "e43b0f4014c44fe5860cc39717d9d1d6",
    #                                     report_name = report_file_path)
    # backtest.run()

if __name__ == "__main__":
    main()
