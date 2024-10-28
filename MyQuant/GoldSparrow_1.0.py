# %% [markdown]
# 获取最新的数据：
# - 更新日期：2024.10.16
# - A股全量数据
# - 目录地址： **~/project/qlib/qlib/data**
# 
# wget https://github.com/chenditc/investment_data/releases/download/2024-10-16/qlib_bin.tar.gz
# 
# tar -zxvf qlib_bin.tar.gz -C ~/project/qlib/qlib/data/cn_data --strip-components=1

# %%
import qlib
import pandas as pd
from qlib.constant import REG_CN
from qlib.utils import  init_instance_by_config
#from qlib.utils import exists_qlib_data
from qlib.workflow import R # 实验记录管理器
from qlib.workflow.record_temp import SignalRecord,SigAnaRecord,PortAnaRecord
from qlib.utils import flatten_dict
from qlib.contrib.report import analysis_position, analysis_model
import quantstats as qs 

from qlib.data import D # 基础行情数据服务的对象

# %% [markdown]
# # 核心实验工作流步骤  
# 本例中训练、预测、回测三个步骤分成三个实验:train, predict,backtest   
#  1 训练机器学习模型  
#  2 预测  
#  3 回测  

# %% [markdown]
# # 1 训练：在训练集和验证集训练模型

# %% [markdown]
# ## 初始化

# %%

# provider_uri = "~/.qlib/qlib_data/cn_data"  # ~表示系统默认的用户目录，通常是C盘Users目录下用户登录名子目录
provider_uri = "~/project/qlib/qlib/data/cn_data"  # 原始行情数据存放目录

# 初始化, kernels=1，在计算特征表达式时只用一个核心，方便计算出错时进行调试。
# qlib.init(provider_uri=provider_uri, region=REG_CN, kernels=1)
qlib.init(provider_uri=provider_uri, region=REG_CN) # 初始化

D.features(['sh000300'], fields=['$open','$high', '$low','$close','$change','$factor','$volume'])

# %% [markdown]
# ## 实例化机器学习模型对象和因子数据集（含因子和标签）

# %%
# 定义股票池。
# stockpool的定义形式举例：stockpool='csi300', stockpool=D.instruments(market='csi300'), 
# stockpool=['sh600000', 'sz000001'] 
stockpool =  D.instruments(market='csi500')

benchmark = "SH000300"  # 基准：沪深300指数
train_start, train_end = "2005-01-01", "2020-12-31" # 训练集
valid_start, valid_end = "2021-01-01", "2022-12-31" # 验证集
test_start, test_end = "2023-01-01", "2024-10-15" # 测试集

start_time = train_start
end_time = test_end
fit_start_time = train_start
fit_end_time = train_end

###################################
# 参数配置
###################################
# 数据处理器参数配置：整体数据开始结束时间，训练集开始结束时间，股票池
data_handler_config = {
    "start_time": start_time,
    "end_time": end_time,
    "fit_start_time": fit_start_time,
    "fit_end_time": fit_end_time,
    "instruments": stockpool,    
}

# 任务参数配置
task = {
    # 机器学习模型参数配置
    "model": {  
        # 模型类
        "class": "LGBModel",
        # 模型类所在模块
        "module_path": "qlib.contrib.model.gbdt", 
        # 模型类超参数配置，未写的则采用默认值。这些参数传给模型类
        "kwargs": {  # kwargs用于初始化上面的class
            "loss": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.0421,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 20,
            "early_stopping_rounds": 50, # 训练迭代提前停止条件
            "num_boost_round": 1000, # 最大训练迭代次数
        },
    },
    "dataset": {  #　因子数据集参数配置
        # 数据集类，是Dataset with Data(H)andler的缩写，即带数据处理器的数据集
        "class": "DatasetH",
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
                ## round 1： 和原始的example表现一致
                #"train": ("2008-01-01", "2014-12-31"),
                #"valid": ("2015-01-01", "2016-12-31"),
                #"test": ("2017-01-01", "2020-08-01"),
                
                ## round 2： 使用最新的数据集
                #"train": ("2008-01-01", "2018-12-31"), # 此时段的数据为训练集
                #"valid": ("2019-01-01", "2022-12-31"), # 此时段的数据为验证集
                #"test": ("2023-01-01", "2024-10-15"),  # 此时段的数据为测试集
                
                ## round 3： 微调validation set,适配特定时间段内曲线特质
                "train": (train_start, train_end), # 此时段的数据为训练集
                "valid": (valid_start, valid_end), # 此时段的数据为验证集
                "test": (test_start, test_end),  # 此时段的数据为测试集
            },
        },
    },
}

# 实例化模型对象
model = init_instance_by_config(task["model"])
# 实例化数据集，从基础行情数据计算出的包含所有特征（因子）和标签值的数据集。
dataset = init_instance_by_config(task["dataset"]) # 类型DatasetH

# %% [markdown]
# ## 执行训练模型实验

# %%
# R变量可以理解为实验记录管理器。
with R.start( experiment_name="train"): # 注意，设好实验名
    # 可选：记录task中的参数到运行记录下的params目录
    R.log_params(**flatten_dict(task))

    # 训练模型，得到训练好的模型model
    model.fit(dataset)
    
    # 可选：训练好的模型以pkl文件形式保存到本次实验运行记录目录下的artifacts子目录，以备后用  
    R.save_objects(**{"trained_model.pkl": model})



    # 打印本次实验记录器信息，含记录器id，experiment_id等信息
    print('info', R.get_recorder().info)

    
    ########################################################################################################
    #                                  说明：
    # 一个实验（比如本实验train）对应mlruns下的一个实验id目录，例如1。
    # 一个实验的每次运行，会在该目录下生成一个不同的实验运行记录id子目录，例如65821e2597014122979f32fef465719f  
    # 运行记录id目录中最重要的子目录是制品目录artifacts，里头保存了实验结果pkl文件
    #########################################################################################################

# %% [markdown]
# # 2 预测：在测试集test上进行预测

# %% [markdown]
# ## 执行预测实验

# %%
with R.start(experiment_name="predict"):
 
    # 当前实验的实验记录器：预测实验记录器
    predict_recorder = R.get_recorder()

    # 生成预测结果文件: pred.pkl, label.pkl存放在运行记录目录下的artifacts子目录   
    # 本实验默认是站在t日结束时刻，预测t+2日收盘价相对t+1日的收益率，计算公式为 Ref($close, -2)/Ref($close, -1) - 1    
    sig_rec = SignalRecord(model, dataset, predict_recorder)  # 将训练好的模型、数据集、预测实验记录器传递给信号记录器      
    sig_rec.generate()


    
    # 生成预测结果分析文件，在artifacts\sig_analysis 目录生成ic.pkl,ric.pkl文件
    sigAna_rec = SigAnaRecord(predict_recorder) # 信号分析记录器
    sigAna_rec.generate()

    print('info', R.get_recorder().info)
    ###########################################################################
    #              说明
    # 由于定义了一个新实验名predict，所以mlruns目录中会新建一个实验id目录，例如2 
    ###########################################################################
    

# %% [markdown]
# ## 预测结果查询
# 

# %% [markdown]
# ### 标签值label和预测值score

# %%
# label_df = predict_recorder.load_object("label.pkl") # 这个pkl文件记录的是测试集未经数据预处理的原始标签值
# 测试集标签值，默认这是经过数据预处理比如标准化处理的（推理数据集的测试集部分）标签值
label_df = dataset.prepare("test", col_set="label") 
label_df.columns = ['label'] # 修改列名LABEL0为label

pred_df = predict_recorder.load_object("pred.pkl") # 加载测试集预测结果到dataframe

print('label_df', label_df) # 预处理后的测试集标签值 
print('pred_df', pred_df) # 测试集对标签的预测值，score就是预测值

# %% [markdown]
# ### 信息系数IC和排序信息系数Rank IC
# 信息系数ic是预测值和标签间的相关系数。每天生成一个信息系数，也就是根据该日所有股票的预测值和标签值计算二者在该日的相关系数。  
# rank IC：每天计算一个所有股票预测值排名与标签排名之间的相关系数，为该日的排序信息系数
# 

# %%
# 信息系数：每天根据所有股票的预测值和标签值，计算出二者在该日的相关系数，即为该日信息系数
ic_df = predict_recorder.load_object("sig_analysis/ic.pkl") 
# 排序信息系数 rank ic：每天根据所有股票的预测值的排名和标签值的排名，计算出二者在该日的排序相关系数，即为该日排序信息系数
print('ic_df', ic_df)
ric_df = predict_recorder.load_object("sig_analysis/ric.pkl") 
print('ric_df', ric_df)

# %% [markdown]
# ### IC均值和Rank IC均值，以及ICIR，Rank ICIR

# %%
print('list_metrics', predict_recorder.list_metrics()) # 所有绩效指标
print('IC', predict_recorder.list_metrics()['IC']) # IC均值：每日IC的均值，一般认为|IC|>0.03说明因子有效，注意 -0.05也认为有预测效能，说明负相关显著
print('ICIR', predict_recorder.list_metrics()['ICIR']) #IC信息率：平均IC/每日IC标准差,也就是方差标准化后的ic均值，一般而言，认为|ICIR|>0.6,因子的稳定性合格
print('Rank IC', predict_recorder.list_metrics()['Rank IC']) # 排序IC均值，作用类似IC
print('Rank ICIR', predict_recorder.list_metrics()['Rank ICIR']) # 排序IC信息率，作用类似ICIR# 此图用于评价因子单调性，组1是因子值最高的一组，组5是因子值最低的一组。

# 这里是评价的是score这个综合因子的有效性和稳定性
# 一般认为|IC|>0.03说明因子有效，|ICIR|>0.6,说明因子稳定

# %% [markdown]
# ## 预测绩效分析图

# %% [markdown]
# ### 准备数据：测试集"预测值"和“标签值”对照表
# 

# %%

# 创建测试集"预测"和“标签”对照表
pred_label_df = pd.concat([pred_df, label_df], axis=1, sort=True).reindex(label_df.index)
#pred_label_df


# %% [markdown]
# ### 信息系数ic 和 rank ic 图  （按天）
# ic 信息系数：预测和标签的相关系数（按天）  
# rank ic 排序信息系数

# %%


analysis_position.score_ic_graph(pred_label_df)
# ic图形横坐标按天显示该日所有股票预测值和标签的相关系数
# 有时候，二者正相关，即预测值越大，则标签值也越大；预测越小，标签也越小。有时负相关，即预测越大，标签越小。有时相关性很小（相关系数接近0）。

# %% [markdown]
# ### 预测模型绩效图

# %%
analysis_model.model_performance_graph(pred_label_df)
# 评价score这个综合因子，以下所说因子指score这个因子

# cumulative Return图
# 用于评价因子单调性，组1是因子值最高的一组，组5是因子值最低的一组。
# 若因子越大的组，收益率越高，说明因子单调性好，也就证明因子对收益率的预测越有效
# 各组收益率差异越大，说明因子特异性高，因子有效。一般看组1和组5的收益率差异是否大即可。

# IC分布图和 IC Normal Dist.Q-Q图
# 观察IC分布是否接近正太分布，越接近正太分布，说明因子越可靠。若ic均值挺大的，但是IC分布图极度但极度的尖峰或右偏，这样的情况，说明因子不可靠。

# Atuo Correlation图
# 评价因子自相关性
# 因子越是具有正的自相关性，则换手率越低，手续费也就越低，默认显示的是lag滞后一期的相关系数
# 如果因子自相关为0或负，则股票今天因子高，明天很可能因子就低，这样造成的结果就是，我们对这个股票，一会儿卖，一会儿买，从而造成很高的手续费。





# %%
analysis_model.model_performance_graph(pred_label_df, N=6,
    graph_names=["group_return", "pred_ic", "pred_autocorr",  "pred_turnover"], 
    rank=True, lag=1, reverse=False, show_notebook=True) # N分几组,lag 自相关图滞后期

# top bottom turnover图
# 展示了1组（top）和5组（bottom）股票的换手率序列

# %% [markdown]
# ### 模型特征重要性

# %%
# 得到特征重要性系列
feature_importance = model.get_feature_importance()
print(feature_importance)
# feature_importance.plot(figsize=(50, 10))

fea_expr, fea_name = dataset.handler.get_feature_config() # 获取特征表达式，特征名字
# 特征名，重要性值的对照字典
feature_importance = {fea_name[int(i.split('_')[1])]: v for i,v in feature_importance.items()}
#feature_importance

# %% [markdown]
# # 3 回测：在测试集test回测

# %% [markdown]
# ## 执行回测实验

# %%
# 回测所需参数配置
port_analysis_config = {   
    "executor": {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
            "verbose": True, # 是否打印订单执行记录
        },
    },    
    "strategy": { # 回测策略相关超参数配置
        "class": "TopkDropoutStrategy",   # 策略类名称  
        "module_path": "qlib.contrib.strategy.signal_strategy", 
        "kwargs": {
            # "model": model, # 模型对象
            # "dataset": dataset, # 数据集
            "signal": (model, dataset), # 信号，也可以是pred_df，得到测试集的预测值score
            "topk": 50,
            "n_drop": 5,
            "only_tradable": True,
            "risk_degree": 0.95, # 资金使用比率
            "hold_thresh": 1, # 股票最小持有天数,默认1天
           

            
        },
    },

    "backtest": { # 回测数据参数
        "start_time": test_start,  # test集开始时间
        "end_time": test_end,  # test集结束时间 
        "account": 10000000,
        "benchmark": benchmark, # 基准
        "exchange_kwargs": {
            "freq": "day", # 使用日线数据
            "limit_threshold": 0.095, # 涨跌停板幅度
            "deal_price": "close",  # 以收盘价成交
            "open_cost": 0.0005, # 开仓佣金费率
            "close_cost": 0.0015, # 平仓佣金费率
            "min_cost": 5, # 一笔交易的最小成本
            #"impact_cost": 0.01, # 冲击成本费率，比如因滑点产生的冲击成本
            "trade_unit": 100, # 对应复权前的交易量为100的整数倍
            
        },
    },
}
# 实验名“backtest”
with R.start(experiment_name="backtest"):

    # 创建组合分析记录器，其中predict_recorder把预测值带进来.
    # pa_rec是组合分析记录器portfolio analysis recorder的缩写。
    pa_rec = PortAnaRecord(predict_recorder, port_analysis_config, "day") 
    # 回测与分析：通过组合分析记录器，在测试集上执行策略回测，并记录分析结果到多个pkl文件,
    # 保存到predict_recorder对应目录的子目录artifacts\portfolio_analysis
    pa_rec.generate() 
    
    print('predict_recorder.experiment_id', predict_recorder.experiment_id, 'predict_recorder.id', predict_recorder.id)
    print('info', R.get_recorder().info) # 本次实验信息

# %% [markdown]
# ## 回测结果查询

# %%
# 回测结果提取到df
indicators_normal_1day_df = predict_recorder.load_object("portfolio_analysis/indicators_normal_1day.pkl")
# indicators_normal_1day_obj_df = predict_recorder.load_object("portfolio_analysis/indicators_normal_1day_obj.pkl")

indicator_analysis_1day_df = predict_recorder.load_object("portfolio_analysis/indicator_analysis_1day.pkl")
port_analysis_1day_df = predict_recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
positions_normal_1day_df = predict_recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
report_normal_1day_df = predict_recorder.load_object("portfolio_analysis/report_normal_1day.pkl")


# %%
print('indicator_analysis_1day_df', indicator_analysis_1day_df)
#
#指标含义参考 https://qlib.readthedocs.io/en/latest/reference/api.html 
#https://github.com/microsoft/qlib/blob/main/qlib/contrib/evaluate.py
#pa is the price advantage in trade indicators
#pos is the positive rate in trade indicators
#ffr is the fulfill rate in trade indicators
#tfr is the trade frequency rate in trade indicators
print('indicators_normal_1day_df \n', indicators_normal_1day_df)
# print('indicators_normal_1day_obj_df \n', indicators_normal_1day_obj_df)

# %%
from pprint import pprint
print('port_analysis_1day_df')
pprint(port_analysis_1day_df)

print('report_normal_1day_df')
pprint( report_normal_1day_df)

# print('positions_normal_1day_df')
# pprint( positions_normal_1day_df)

# %% [markdown]
# ## 回测绩效分析图

# %% [markdown]
# ### 收益率图
# 累积收益率，收益率最大回撤，累积超额收益率，累积超额收益率最大回测，换手率

# %%
# 回测结果分析图


analysis_position.report_graph(report_normal_1day_df)

# %% [markdown]
# ### 风险分析图  
# 超额收益率标准差，年化超额收益率，超额收益率信息率information_ratio，超额收益率最大回撤
# 

# %%
analysis_position.risk_analysis_graph(port_analysis_1day_df, report_normal_1day_df)

# %% [markdown]
# #  dataset数据查询：特征，标签

# %% [markdown]
# ## 查看全部特征和标签数据

# %%
# df_test =  dataset.prepare(segments=["test"], data_key = "raw") 
# 返回（原始数据集中）训练集、验证集、测试集的全部特征和标签数据
df_train, df_valid, df_test =  dataset.prepare(segments=["train", "valid", "test"], data_key = "raw") 

#df_test

# %% [markdown]
# ## 查看标签（即预测对象）的定义

# %%
label_expr, label_name = dataset.handler.get_label_config()
print('label_expr',label_expr)
print('label_name',label_name)

# %% [markdown]
# 
# ## 查看特征定义

# %%

fea_expr, fea_name = dataset.handler.get_feature_config()
print('fea_expr',fea_expr)
print()
print('fea_name',fea_name)


# %% [markdown]
# # dataset保存为pickle文件
# 

# %%
dataset.config(dump_all=True, recursive=True)
dataset.to_pickle(path="dataset.pkl", dump_all=True)

# %% [markdown]
# # pyfolio和quantstats对回测绩效的评价

# %%
df_benchmark = D.features(['sh000300'], fields=['$close','$change','$factor'])
#df_benchmark

df_benchmark_returns = df_benchmark.loc['sh000300']['$close'].pct_change()
#df_benchmark_returns

df_benchmark['origClose'] = df_benchmark['$close'] /  df_benchmark['$factor']
#df_benchmark

# %%


# 使用quantstats输出绩效

returns = report_normal_1day_df['return']

##
# fixed issue: Monthly returns are not calculated correctly
# detailed information: https://github.com/ranaroussi/quantstats/issues/255
import locale    
old_locale=locale.getlocale(locale.LC_ALL)
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
returns['Month'] = returns.index.strftime('%b')
locale.setlocale(locale.LC_ALL, old_locale)

pprint(returns)
file_path = "/home/godlike/project/qlib/qlib/experiment_data/qs_report3.html"
qs.reports.html(returns, benchmark = df_benchmark_returns.loc[returns.index], output=file_path)

