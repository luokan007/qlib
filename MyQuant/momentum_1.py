## 第一个策略
## 目标：
##      1. 复现Quantitative Momentum书中的策略：
##          * 沪深300
##          * 动量筛查并排序：根据过去12个月的收益，将计算一般动量，筛选头部50只
##          * 动量质量筛查并排序，筛选头部25只
##          * 动量季节性筛查，每个季度再平衡
##          飞书：https://k112xm59y5.feishu.cn/wiki/VMPuwZYuSiq8ZSkABFccqznknpd
##      2. 完整走通qlib的流程
import qlib
import pandas as pd
