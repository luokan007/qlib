# title: intraday_benchmark.py
# updated: 2024.12.4
# change log:
#   - 构建基于backtrader的回测框架
#   - 

# 目标：
#   1. 载入qlib数据，载入日频数据
#   2. 支持backtrader框架
#   3. 以开盘价买入
#   4. 增加滑点、
#   5. 改进risk_degree的计算方式


import quantstats as qs

from datetime import datetime, time
from datetime import timedelta
import math
import pandas as pd
import numpy as np
import backtrader as bt
import os.path  # 管理路径
import sys  # 发现脚本名字(in argv[0])
import glob
from backtrader.feeds import PandasData  # 用于扩展DataFeed
import qlib
from qlib.workflow import R
from qlib.data import D  # 基础行情数据服务的对象

def main(provider_uri=None, exp_name=None, rid=None, pred_score_df=None):
    
  
    import webbrowser
    webbrowser.open(output)
    print('耗时',datetime.now()-starttime)

if __name__ == "__main__":
    ##### pred时间段为2023-01-01 至2023-01-30,主要为了测试流程  rid: "0833139cd23a48d592f1a1c6510f8495"
    ##### pred时间段为2023-01-01 至2024-10-30,形成结论  rid: "156de12d5bd8429882e24c11f5593a5b"
    ### pred时间段为2023-01-01 至2024-10-30, ALSTM模型，  rid: 57c61d4d74314018abe86204df221a34

    main(provider_uri=r"/home/godlike/project/GoldSparrow/Updated_Stock_Data",
         exp_name="LSTM_CSI300_Alpha58",
         rid="0833139cd23a48d592f1a1c6510f8495" ##"7c5183bbecbc4ebd95828de1784def47"
         )