# title: GoldSparrow_1.9.py
# updated: 2024.10.24
# change log:
#   - 支持Transformer


import pickle
import os
import locale

import quantstats as qs 
import qlib

from qlib.constant import REG_CN
from qlib.utils import  init_instance_by_config #qlib初始化函数
#from qlib.utils import exists_qlib_data
from qlib.workflow import R # 实验记录管理器
from qlib.workflow.record_temp import SignalRecord,SigAnaRecord,PortAnaRecord #实验管理器

### 暂时还用不到的包，注释掉
#from qlib.backtest import backtest, executor
#from qlib.contrib.evaluate import risk_analysis
#from qlib.contrib.strategy import TopkDropoutStrategy
#from qlib.utils import flatten_dict
#from qlib.utils.time import Freq
#from qlib.contrib.report import analysis_position, analysis_model # 模型分析

from qlib.data import D # 基础行情数据服务的对象

global_config = {
    "qlib_init": {
        "provider_uri":  "/home/godlike/project/GoldSparrow/Updated_Stock_Data"  # 原始行情数据存放目录
    },
    "market": 'csi300',  # 股票池
    "benchmark": "SH000300", # 基准：沪深300指数
    "train_start": "2005-01-01",
    "train_end": "2020-12-31", # 训练集
    "valid_start": "2021-01-01", 
    "valid_end": "2022-12-31", # 验证集
    "test_start": "2023-01-01", 
    "test_end": "2024-10-30",  # 测试集
    "dataset_pickle_path": "~/project/qlib/qlib/experiment_data/dataset.1.9.pkl",
    "train_model_pickle_path":"~/project/qlib/qlib/experiment_data/train_model.1.9.pkl",
    "qs_report_file_path":"~/project/qlib/qlib/experiment_data/qs_report.1.9.html"
}

def training_process(dataset):
    """_summary_
    - 执行训练过程
    - 导出数据文件和模型文件到pickle
    - 输出简要的模型评估结果
    Returns:
        None
    """
    # 任务参数配置
    model_config = {
        # 机器学习模型参数配置
        # 模型类
        "class": "TransformerModel",
        # 模型类所在模块
        "module_path": "qlib.contrib.model.pytorch_transformer_ts", 
        # 模型类超参数配置，未写的则采用默认值。 这些参数传给模型类
        "kwargs": {  # kwargs用于初始化上面的class
             "d_feat": 158,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0,
            "n_epochs":  100,
            "lr": 1e-4,
            "early_stop": 10,
            "batch_size": 800,
            "metric": "loss",
            "loss": "mse",
            "n_jobs": 8,
            "GPU": 0,
        }
    }

    # 实例化模型对象
    model = init_instance_by_config(model_config)
    # 训练模型
    print("Training Model...")
    with R.start( experiment_name="train"): # 注意，设好实验名
        
        model.fit(dataset)
        # 可选：训练好的模型以pkl文件形式保存到本次实验运行记录目录下的artifacts子目录，以备后用  
        R.save_objects(**{"trained_model.pkl": model})
        # 打印本次实验记录器信息，含记录器id，experiment_id等信息
        print('info', R.get_recorder().info)
        print("Model Training Done!")
    
    
    #将模型，数据集保存为pickle文件以备后用
    param_model_pickle_path = global_config["train_model_pickle_path"]
    
    # 扩展波浪线路径
    if param_model_pickle_path.startswith("~"):
        _tmp_file_path = os.path.expanduser(param_model_pickle_path)
    else:
        _tmp_file_path = param_model_pickle_path
    
    # 获取文件所在的目录
    _tmp_directory_path = os.path.dirname(_tmp_file_path)
    
    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(_tmp_directory_path):
        os.makedirs(_tmp_directory_path)

    model.config(dump_all=True, recursive=True)
    model.to_pickle(path=_tmp_file_path, dump_all=True)
    print("Output pickle file Done!")

    print("\nTraining process Done!")
    return model

def pred_backtest_process(dataset,model) -> None:
    """
     - 从pickle文件载入模型和数据文件
     - 回测
    """
    
    ####
    param_benchmark = global_config["benchmark"]
    
    param_test_start = global_config["test_start"]
    param_test_end = global_config["test_end"]
    
    FREQ = "day" # 使用日线数据
    param_account = 10000000 # 账户金额，1千万
    
    port_analysis_config = {   
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": FREQ,
                "generate_portfolio_metrics": True,
                # "verbose": True, # 是否打印订单执行记录
            },
        },    
        "strategy": { # 回测策略相关超参数配置
            "class": "TopkDropoutStrategy",   # 策略类名称  
            "module_path": "qlib.contrib.strategy.signal_strategy", 
            "kwargs": {
                # "model": model, # 模型对象
                # "dataset": dataset, # 数据集
                "signal": (model, dataset), 
                "topk": 50,
                "n_drop": 5,
                "only_tradable": True,
                "risk_degree": 0.95, # 资金使用比率
                "hold_thresh": 1, # 股票最小持有天数,默认1天
            },
        },

        "backtest": {
            "start_time": param_test_start,  # test集开始时间 
            "end_time": param_test_end,  
            "account": param_account,
            "benchmark": param_benchmark, # 基准
            "exchange_kwargs": {
                "freq": "day", # 使用日线数据
                "limit_threshold": 0.095, # 涨跌停板幅度
                "deal_price": "close",  # 以收盘价成交
                "open_cost": 0.0005, # 开仓佣金费率
                "close_cost": 0.0015, # 平仓佣金费率
                "min_cost": 5, # 一笔交易的最小成本
                "trade_unit": 100, # 对应复权前的交易量为100的整数倍
                
            },
        },
    }
    
        
    with R.start(experiment_name="LSTM_CSI300_Alpha58"):
        # 当前实验的实验记录器：预测实验记录器
        predict_recorder = R.get_recorder()
        print('predict_recorder.experiment_id', predict_recorder.experiment_id, 'predict_recorder.id', predict_recorder.id)
        print('info', R.get_recorder().info) # 本次实验信息

        # 生成预测结果文件: pred.pkl, label.pkl存放在运行记录目录下的artifacts子目录   
        # 本实验默认是站在t日结束时刻，预测t+2日收盘价相对t+1日的收益率，计算公式为 Ref($close, -2)/Ref($close, -1) - 1    
        sig_rec = SignalRecord(model, dataset, predict_recorder)  # 将训练好的模型、数据集、预测实验记录器传递给信号记录器      
        sig_rec.generate()

        # 生成预测结果分析文件，在artifacts\sig_analysis 目录生成ic.pkl,ric.pkl文件
        sigAna_rec = SigAnaRecord(predict_recorder) # 信号分析记录器
        sigAna_rec.generate()
        print('info', R.get_recorder().info)
        
        pa_rec = PortAnaRecord(predict_recorder, port_analysis_config, "day") 
        # 回测与分析：通过组合分析记录器，在测试集上执行策略回测，并记录分析结果到多个pkl文件,
        # 保存到predict_recorder对应目录的子目录artifacts\portfolio_analysis
        pa_rec.generate()
        
        
    
    # label_df = predict_recorder.load_object("label.pkl") # 这个pkl文件记录的是测试集未经数据预处理的原始标签值
    # 测试集标签值，默认这是经过数据预处理比如标准化处理的（推理数据集的测试集部分）标签值
    label_df = dataset.prepare("test", col_set="label") 
    label_df.columns = ['label'] # 修改列名LABEL0为label

    pred_df = predict_recorder.load_object("pred.pkl") # 加载测试集预测结果到dataframe
    # 创建测试集"预测"和“标签”对照表
    #pred_label_df = pd.concat([pred_df, label_df], axis=1, sort=True).reindex(label_df.index)

    # 信息系数：每天根据所有股票的预测值和标签值，计算出二者在该日的相关系数，即为该日信息系数
    ic_df = predict_recorder.load_object("sig_analysis/ic.pkl") 
    # 排序信息系数 rank ic：每天根据所有股票的预测值的排名和标签值的排名，计算出二者在该日的排序相关系数，即为该日排序信息系数
    print('ic_df', ic_df)
    ric_df = predict_recorder.load_object("sig_analysis/ric.pkl") 
    print('ric_df', ric_df)
    print('list_metrics', predict_recorder.list_metrics()) # 所有绩效指标
    print('IC', predict_recorder.list_metrics()['IC']) # IC均值：每日IC的均值，一般认为|IC|>0.03说明因子有效，注意 -0.05也认为有预测效能，说明负相关显著
    print('ICIR', predict_recorder.list_metrics()['ICIR']) #IC信息率：平均IC/每日IC标准差,也就是方差标准化后的ic均值，一般而言，认为|ICIR|>0.6,因子的稳定性合格
    print('Rank IC', predict_recorder.list_metrics()['Rank IC']) # 排序IC均值，作用类似IC
    print('Rank ICIR', predict_recorder.list_metrics()['Rank ICIR']) # 排序IC信息率，作用类似ICIR# 此图用于评价因子单调性，组1是因子值最高的一组，组5是因子值最低的一组。
       
        
    # 回测结果提取到df
    report_normal_1day_df = predict_recorder.load_object("portfolio_analysis/report_normal_1day.pkl")

    
    # 使用quantstats输出绩效
    returns = report_normal_1day_df['return']
    df_benchmark = D.features(['sh000300'], fields=['$close','$change','$factor'])
    df_benchmark_returns = df_benchmark.loc['sh000300']['$close'].pct_change()
    df_benchmark['origClose'] = df_benchmark['$close'] /  df_benchmark['$factor']
    
    #returns
    param_report_html_path = global_config["qs_report_file_path"]
    # 扩展波浪线路径
    if param_report_html_path.startswith("~"):
        _tmp_file_path = os.path.expanduser(param_report_html_path)
    else:
        _tmp_file_path = param_report_html_path
    
    qs.reports.html(returns, benchmark = df_benchmark_returns.loc[returns.index], output=_tmp_file_path, rf=0.0)
        
    return None

def process_data():
    """_summary_

    Returns:
        _type_: _description_
    """
    
    ##配置训练参数    
    param_market = global_config["market"]
    param_stockpool =  D.instruments(market=param_market)
    
    start_time = global_config["train_start"]
    end_time = global_config["test_end"]
    fit_start_time = global_config["train_start"]
    fit_end_time = global_config["valid_end"]
    
    train_start = global_config["train_start"]
    train_end = global_config["train_end"]
    valid_start = global_config["valid_start"]
    valid_end = global_config["valid_end"]
    test_start = global_config["test_start"]
    test_end = global_config["test_end"]
    
    # 数据处理器参数配置：整体数据开始结束时间，训练集开始结束时间，股票池
    data_handler_config = {
        "start_time": start_time,
        "end_time": end_time,
        "fit_start_time": fit_start_time,
        "fit_end_time": fit_end_time,
        "instruments": param_stockpool,   
        
        "infer_processors":[
            #{
            #    "class": "FilterCol",
            #    "kwargs":{
            #        "fields_group": "feature",
            #        "col_list": ["RESI5", "WVMA5", "RSQR5", "KLEN", 
            #                     "RSQR10", "CORR5", "CORD5", "CORR10", 
            #                     "ROC60", "RESI10", "VSTD5", "RSQR60",
            #                     "CORR60", "WVMA60", "STD5", "RSQR20",
            #                     "CORD60", "CORD10", "CORR20", "KLOW"]
            #        }
            #},
            {
                "class": "RobustZScoreNorm",
                "kwargs":{
                    "fields_group": "feature",
                    "clip_outlier": "true"}
            },
            {
                "class": "Fillna",
                "kwargs":{"fields_group": "feature"}
            },
        ],
         
        "learn_processors":[
            {
                "class": "DropnaLabel"
            },
            {
                "class": "CSRankNorm",
                "kwargs":{"fields_group": "label"}
            },
        ],
        "label": [["Ref($close, -2) / Ref($close, -1) - 1"],["LABEL0"]]
    }
    
    
    dataset_config = {  #　因子数据集参数配置
        # 数据集类，是Dataset with Data(H)andler的缩写，即带数据处理器的数据集
        "class": "TSDatasetH",
        # 数据集类所在模块
        "module_path": "qlib.data.dataset",
        # 数据集类的参数配置
        "kwargs": { 
            "handler": { # 数据集使用的数据处理器配置
                "class": "Alpha158", # 数据处理器类，继承自DataHandlerLP
                "module_path": "qlib.contrib.data.handler", # 数据处理器类所在模块
                "kwargs": data_handler_config, # 数据处理器参数配置
            },
            "segments": { # 数据集时段划分              
                "train": (train_start, train_end), # 此时段的数据为训练集
                "valid": (valid_start, valid_end), # 此时段的数据为验证集
                "test": (test_start, test_end),  # 此时段的数据为测试集
            },
            "step_len":20
        },
    }
    # 实例化因子库数据集，从基础行情数据计算出的包含所有特征（因子）和标签值的数据集。
    _dataset = init_instance_by_config(dataset_config) # 返回DatasetH类型
    param_dataset_pickle_path = global_config["dataset_pickle_path"]
    
    # 扩展波浪线路径
    if param_dataset_pickle_path.startswith("~"):
        _tmp_file_path = os.path.expanduser(param_dataset_pickle_path)
    else:
        _tmp_file_path = param_dataset_pickle_path
    
    
    # 获取文件所在的目录
    _tmp_directory_path = os.path.dirname(_tmp_file_path)
    
    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(_tmp_directory_path):
        os.makedirs(_tmp_directory_path)
    
    _dataset.config(dump_all=True, recursive=True)
    _dataset.to_pickle(path=_tmp_file_path, dump_all=True)
    return _dataset

def main(is_process_data = True, is_train =True):
    """_summary_

    Args:
        is_process_data (bool, optional): _description_. Defaults to True.
        is_train (bool, optional): _description_. Defaults to True.
    """
    ##初始化
    param_provider_uri = global_config["qlib_init"]["provider_uri"]
    qlib.init(provider_uri=param_provider_uri, region=REG_CN) # 初始化
   
    if is_process_data:
        _main_dataset = process_data()
    else:
        param_dataset_pickle_path = global_config["dataset_pickle_path"]
        # 扩展波浪线路径
        if param_dataset_pickle_path.startswith("~"):
            _tmp_file_path = os.path.expanduser(param_dataset_pickle_path)
        else:
            _tmp_file_path = param_dataset_pickle_path
        
        with open(_tmp_file_path, "rb") as file_dataset:
            _main_dataset = pickle.load(file_dataset)
    
    if(is_train):
        _main_model = training_process(_main_dataset)
    else:
        param_train_model_path = global_config["train_model_pickle_path"]
        # 扩展波浪线路径
        if param_train_model_path.startswith("~"):
            _tmp_file_path = os.path.expanduser(param_train_model_path)
        else:
            _tmp_file_path = param_train_model_path
        
        with open(_tmp_file_path, "rb") as file_model:
            _main_model = pickle.load(file_model)
        
    pred_backtest_process(_main_dataset,_main_model)


########
    # fixed issue: 中文环境会导致Monthly Return表格展示全部为零，具体可见下方链接
    # 修复方法：将locale设置为en_US.UTF-8
    # detailed information: https://github.com/ranaroussi/quantstats/issues/255
    ##
    ##  import os
    ##  import locale
# 临时设置语言环境为 en_US.UTF-8
os.environ['LC_ALL'] = 'en_US.UTF-8'
os.environ['LANG'] = 'en_US.UTF-8'
# 确保设置生效
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
########
if __name__ == "__main__":
    main(is_process_data = True, is_train =True)