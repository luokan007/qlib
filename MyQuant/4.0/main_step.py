# %%
import qlib
import pandas as pd
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R # 实验记录管理器
from qlib.workflow.record_temp import SignalRecord,SigAnaRecord,PortAnaRecord
from qlib.utils import flatten_dict

from qlib.data.dataset.loader import QlibDataLoader, StaticDataLoader
from qlib.data import D # 基础行情数据服务的对象
from qlib.data.filter import ExpressionDFilter
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import CSZScoreNorm, DropnaProcessor, RobustZScoreNorm, Fillna,DropnaLabel,CSRankNorm
from qlib.data.dataset import DatasetH
from qlib.data.dataset import TSDatasetH
from qlib.contrib.model.linear import LinearModel
from qlib.contrib.model.pytorch_lstm import LSTM
from qlib.contrib.model.pytorch_alstm_ts import ALSTM
from qlib.contrib.data.handler import MyAlpha158Ext
from qlib.contrib.data.handler import Alpha158


# %% [markdown]
# # 第一步
# ## 定义股池及时间 
# ## 计算加载因子与标签，执行因子与标签预处理
# ## 生成模型所需数据集（含时段划分信息）

# %%
# qlib内置行情数据存放目录
provider_uri = "/home/godlike/project/GoldSparrow/Day_Data/Day_data/qlib_data"  
qlib.init(provider_uri=provider_uri, region="cn")
# 股池设置
pool = 'csi300'

# 时段设置
# train = ('2008-01-01','2014-12-31')
# valid = ('2015-01-01','2016-12-31')
# test = ('2017-01-01', '2020-12-31')

train = ('2008-01-01','2016-12-31')
valid = ('2017-01-01','2019-12-31')
test = ('2020-01-01', '2024-12-18')
model_type = 'lstm' # 模型类型


# 要采用的机器学习模型是否有早停机制
early_stopping = True 
if not early_stopping: # 如果没有早停机制，则无需验证集，故扩展训练集包含验证集，以免浪费数据       
    # 在qlib目录中搜索early_stop，发现CatBoostModel, DEnsembleModel, LGBModel, HFLGBModel, XGBModel含early_stopping_rounds参数，
    # ADARNN, ADD, ALSTM, GATs, GRU, HIST, IGMTF, KRNN, LocalformerModel, LSTM,Sandwich, SFM, TabnetModel,TCN, TCTS, TRAModel, TransformerModel含early_stop参数
    # 以上模型都可以执行早停机制，
    # 其他模型未提供早停接口，如线性回归模型无早停机制，
    train = (train[0], valid[1])

# 整体数据范围
start_time=train[0]
end_time=test[1]

# 因子与标签数据预处理时可能会用到这个时段
fit_start_time = train[0]
fit_end_time = train[1] 


# 推理处理器，RobustZScoreNorm要算fit_start_time和fit_end_time间的因子均值和方差，
# 然后因子要减去均值除以标准差就行正则化
infer_processors = [RobustZScoreNorm(fit_start_time=fit_start_time, fit_end_time=fit_end_time, 
                                     fields_group='feature',
                                     clip_outlier=True),Fillna(fields_group='feature')]

# infer_processors = [RobustZScoreNorm(fit_start_time=fit_start_time, fit_end_time=fit_end_time, 
#                                       fields_group='feature',
#                                       clip_outlier=True),DropnaProcessor(fields_group='feature')]
learn_processors = [DropnaLabel(),CSRankNorm(fields_group='label')]


filter_rule = None # ExpressionDFilter(rule_expression='EMA($close, 10)<10')
data_handler = MyAlpha158Ext(instruments=pool,
        start_time=start_time,
        end_time=end_time,
        freq="day",
        infer_processors=infer_processors,
        learn_processors=learn_processors,
        fit_start_time=fit_start_time,
        fit_end_time=fit_end_time,     
        filter_pipe=filter_rule,
       )

# data_handler = Alpha158(instruments=pool,
#         start_time=start_time,
#         end_time=end_time,
#         freq="day",
#         infer_processors=infer_processors,
#         learn_processors=learn_processors,
#         fit_start_time=fit_start_time,
#         fit_end_time=fit_end_time,     
#         filter_pipe=filter_rule,
#        )

# 最终生成数据集
if model_type == "lstm":
    ds = DatasetH(handler = data_handler, segments={"train": train,"valid": valid, "test":test})
elif model_type == "linear":
    ds = DatasetH(handler = data_handler, segments={"train": train,"test":test})
elif model_type == "alstm":
    ds = TSDatasetH(step_len = 30, handler = data_handler, segments={"train": train,"valid": valid, "test":test})
else:
    raise ValueError(f"model_type={model_type} not supported")

# %%
if early_stopping == False: # 若使用不带早停的模型，如线性回归模型
    from  qlib.contrib.model.linear import LinearModel
    model = LinearModel(estimator="ridge",alpha=0.05)

if early_stopping == True: # 若使用带早停的模型，如线LGBModel
    if model_type == "lstm":
        model = LSTM(loss = "mse", 
                d_feat = 236,
                hidden_size=64,
                num_layers=2,
                dropout=0,
                n_epochs=50,
                lr= 0.00001,
                early_stop=20,
                batch_size=800,
                metric="loss",
                GPU=0)
    elif model_type == "alstm":
        model = ALSTM(d_feat=236,
                hidden_size=64,
                num_layers=2,
                dropout=0,
                n_epochs=50,
                lr= 0.00001,
                early_stop=20,
                batch_size=800,
                metric="loss",
                loss="mse",
                n_jobs=8,
                GPU=0,
                rnn_type="GRU")
    else:
        raise ValueError(f"model_type={model_type} not supported")

# 模型训练, 使用fit方法
model.fit(dataset=ds)

# %%
import os
from pathlib import Path

# 测试集上执行预测
pred_series = model.predict(dataset=ds)

# 转换为DataFrame并添加列名
pred = pred_series.to_frame("score")

# 设置输出目录
output_dir = "/home/godlike/project/GoldSparrow/Temp_Data"
Path(output_dir).mkdir(parents=True, exist_ok=True)

# 格式化索引和导出
pred.index = pred.index.set_names(['datetime', 'instrument'])

# 构建输出路径
csv_path = os.path.join(output_dir, 'pred.csv')
pkl_path = os.path.join(output_dir, 'pred.pkl')

# 保存文件
pred.reset_index().to_csv(csv_path, index=False)
pred.to_pickle(pkl_path)

print(f"预测结果已保存至:\n- {csv_path}\n- {pkl_path}")


## 提取label文件，计算ic/icir/long precision/short precision

params = dict(segments="test", col_set="label", data_key=DataHandlerLP.DK_R)
label = ds.prepare(**params)
print(label)

from eval_model import MyEval
eval = MyEval.from_dataframe(pred, label)
print(eval.eval())




