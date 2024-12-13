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
        conf_path=CONF_LIST[0],
        step=40,  # 滚动步长，每隔40天滚动训练一次，它也决定了每滚测试集长度为40天
        horizon=1,  # 收益率预测期长度
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

if __name__ == "__main__":
    Offline_Evaluate()  ###滚动模型的离线评测
    
# cd /home/godlike/project/GoldSparrow    
# wget https://github.com/chenditc/investment_data/releases/download/2024-11-25/qlib_bin.tar.gz
# 
# tar -zxvf qlib_bin.tar.gz --strip-components=1
# 
    