# title: intraday_1.2.py
# updated: 2024.12.9
# change log:
#   - 统计日内波动
#   - 

# 目标：
#   1. 拟合数据股票一个时段内的正态分布
#   2 基于股票的正太分布，
#       - 买入：开盘价基础上，下调一个标准差的买单，如果不能成交，则以收盘价成交
#       - 卖出：开盘价基础上，上调一个标准差的卖单，如果不能成交，则以收盘价成交

import quantstats as qs
import webbrowser
from datetime import datetime, time
from datetime import timedelta
from scipy import stats
from scipy.stats import beta
import math
import pandas as pd
import numpy as np
import csv
import backtrader as bt
import os.path  # 管理路径
import sys  # 发现脚本名字(in argv[0])
import glob
from backtrader.feeds import PandasData  # 用于扩展DataFeed
import qlib
from qlib.workflow import R
from qlib.data import D  # 基础行情数据服务的对象

class StampDutyCommissionScheme(bt.CommInfoBase):
    '''
    本佣金模式下，买入股票仅支付佣金，卖出股票支付佣金和印花税.    
    '''
    params = (
        ('stamp_duty', 0.0005),  # 印花税率
        ('commission', 0.00025),  # 佣金率
        ('min_cost', 5),        #每单最小交易成本5元
        ('percabs', True),
        ('stocklike', True),
    )

    def _getcommission(self, size, price, pseudoexec):
        '''
        If size is greater than 0, this indicates a long / buying of shares.
        If size is less than 0, it indicates a short / selling of shares.
        '''
        if size > 0:  # 买入，不考虑印花税,最低交易成本5元
            return max(size * price * self.p.commission, self.p.min_cost)
        elif size < 0:  # 卖出，考虑印花税,最低交易成本5元
            return max(-size * price * (self.p.stamp_duty + self.p.commission), self.p.min_cost)
        else:
            return 0  # just in case for some reason the size is 0.

class TopkDropoutStrategy(bt.Strategy):
    params = dict(pred_df_all=None, high_dist=None, low_dist=None, topk=50, n_drop=5, risk_degree=0.90, trade_unit=100)

    # 日志函数
    def log(self, txt, dt=None):
        # 以第一个数据data0，即指数作为时间基准
        dt = dt or self.data0.datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):

        # 记录以往订单，在再平衡日要全部取消未成交的订单
        self.order_list = []
        self.notify_delistdays = 5  # 提前几天知道退市
        self.start_date = self.p.pred_df_all.index[0][0]
        self.end_date = self.p.pred_df_all.index[-1][0]
        self.tomorrow_order_dic = {}  ##k: symbol, v: (direction,amount)
        self.market_open_time = datetime.strptime("09:30", "%H:%M")
        self.market_pre_close_time = datetime.strptime("14:50", "%H:%M")
        self.market_close_time = datetime.strptime("14:55", "%H:%M")
        self.BUY_DIRECTION = 1
        self.SELL_DIRECTION = -1
        self.RESILIANT_RATE = 0.05
        self.verbose=True

    def start(self):
        ### 计算均匀划分的份数
        pass

    def prenext(self):
        self.next()

    def next(self):

        now = self.datetime.datetime(0)
        
        ###在9:30，更新订单列表中的目标价，创建今日有效的限价单
        if now.time() == self.market_open_time.time():
            if len(self.tomorrow_order_dic) > 0:
                for symbol, (direction,amount) in self.tomorrow_order_dic.items():
                    target_data = self.getdatabyname(symbol)
                    open_price = target_data.open[0]
                    target_price = self.update_target_price(symbol,open_price,direction)
                    if direction == self.BUY_DIRECTION:
                        order = self.buy(data=target_data,
                                        size=amount,
                                        exectype=bt.Order.Limit,
                                        price=target_price,
                                        valid=bt.Order.DAY)
                        self.order_list.append((order,target_data,direction,amount))
                    elif direction == self.SELL_DIRECTION:
                        order = self.sell(data=target_data,
                                        size=amount,
                                        exectype=bt.Order.Limit,
                                        price=target_price,
                                        valid=bt.Order.DAY)
                        self.order_list.append((order,target_data,direction,amount))
                    else:
                        raise ValueError("direction should be 1 or -1")
            else:
                pass
        ## 在收盘倒数第二个时段，取消所有未成交订单，创建明天的待成交订单
        elif now.time() == self.market_pre_close_time.time():
            for (order,data,direction,amount) in self.order_list:
                ##如果订单未成交，则取消，并创建市价单成交,此处有逻辑的缺陷：未考虑部分成交的状态(Order.Partial)，因为金额都较小，
                if order.status not in [order.Completed]:
                    self.cancel(order)  # 取消未执行订单
                    if direction == self.BUY_DIRECTION:
                        self.buy(data=data,
                                 size=amount,
                                 exectype=bt.Order.Market)
                    elif direction == self.SELL_DIRECTION:
                        self.sell(data=data,
                                 size=amount,
                                 exectype=bt.Order.Market)
                    else:
                        raise ValueError("direction should be 1 or -1")
            self.order_list = []  # 重置
        elif now.time() == self.market_close_time.time():        
            self.tomorrow_order_dic= self.rebalance_portfolio()  # 执行再平衡
            p_list = [p for p in self.getpositions().values() if p.size != 0]
            print('num position', len(p_list))  # 持仓品种数
        else:
            pass
        
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # 订单状态 submitted/accepted，无动作
            return

        # 订单完成
        if(self.verbose is True):
            if order.status in [order.Completed]:
                if order.isbuy():
                    self.log(
                        f'买单执行,{bt.num2date(order.executed.dt)}, {order.data._name}, {order.executed.price}, {order.executed.size}, 创建时间 {bt.num2date(order.created.dt)}'
                    )

                elif order.issell():
                    self.log(
                        f'卖单执行,{bt.num2date(order.executed.dt)}, {order.data._name}, {order.executed.price}, {order.executed.size}, 创建时间 {bt.num2date(order.created.dt)}'
                    )

            else:
                self.log(
                    f'订单作废  {order.data._name}, {order.getstatusname()}, isbuy:{order.isbuy()}, {order.created.size}, 创建时间 {bt.num2date(order.created.dt)}'
                )

    # 记录交易收益情况
    def notify_trade(self, trade):
        if trade.isclosed:
            self.log('毛收益 %0.2f, 扣佣后收益 % 0.2f, 佣金 %.2f, 市值 %.2f, 现金 %.2f' %
                     (trade.pnl, trade.pnlcomm, trade.commission,
                      self.broker.getvalue(), self.broker.getcash()))

    def get_first_n(self, li, n):
        cur_n = 0
        res = []  # 结果

        for symbol in li:
            d = self.getdatabyname(symbol)  # 取得行情对象
            if len(d) >= d.buflen(
            ) - self.notify_delistdays and d.datetime.date(
                    0) < self.end_date - timedelta(days=2 * self.notify_delistdays):
                # 即将退市的股票不加入买单
                continue
            res.append(symbol)
            cur_n += 1
            if cur_n >= n:
                break
        return res

    def get_last_n(self, li, n):  # 需要平仓的股票池

        cur_n = 0
        res = []  # 结果

        for symbol in reversed(li):
            res.append(symbol)
            cur_n += 1
            if cur_n >= n:
                break
        return res[::-1]        
    
    def update_target_price(self, symbol, price, direction):
        """_summary_
            假设股价涨跌分别满足不同的beta分布,买入的情况下在低点上移动,卖出在高点上移动
        Args:
            symbol (_type_): 股票代码
            price (_type_): 当日开盘价
            direction (_type_): 买入或是卖出

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        ret_price = -1
        if direction == self.BUY_DIRECTION:
            (l_alpha,l_beta,l_loc,l_scale) = self.p.low_dist[symbol]
            x = beta.ppf(self.RESILIANT_RATE, l_alpha,l_beta,l_loc,l_scale)
            ret_price = min( round((1 - x*0.1)*price,2), price-0.02)
        elif direction == self.SELL_DIRECTION:
            (h_alpha,h_beta,h_loc,h_scale) = self.p.high_dist[symbol]
            x = beta.ppf(self.RESILIANT_RATE, h_alpha,h_beta,h_loc,h_scale)
            ret_price = max( round((x*0.1+1)*price,2) , price+0.02)
        else:
            raise ValueError("direction should be 1 or -1")
        return ret_price
        
    def rebalance_portfolio(self):
        def will_delist(symbol):
            # 是否即将退市
            d = self.getdatabyname(symbol)
            # print("qq",type(d.datetime.date(0)),type(self.end_date))
            if len(d) >= d.buflen() - self.notify_delistdays and \
                pd.Timestamp(d.datetime.date(0)) < self.end_date-timedelta(days=2*self.notify_delistdays):
                return True
            else:
                return False
        ret_dic = {}
        
        forbid = [] # 其它禁止持有的股票
        # 取得当前日期
        curr_date = pd.to_datetime(self.datetime.date(0))
        print('curr_date=====', curr_date)
        
        pred_score = self.p.pred_df_all.xs(curr_date)  # 本日各股预测分，df
        to_delist = [
            symbol for symbol in pred_score.index if will_delist(symbol)
        ]  # 股池中即将退市的股票。
        pred_score = pred_score[
            ~pred_score.index.isin(to_delist+forbid)]  # 股池去掉即将退市的股票，其它不想买的股票也可在此删除

        # 获取账户现金值
        cash = self.broker.getcash()

        current_stock_list = [
            d._name for d, p in self.getpositions().items() if p.size != 0
        ]  # 当前有持仓的股票代码集合
        
        ##获得当前累积资产
        total_value = self.broker.getvalue()
        #print(f'Total asset value: {total_value}')
        

        # 今日已持仓股票列表，index类型，已按score降序排列。若某持仓股票不在pred_score中，则该股票排在index最后。
        last = pred_score.reindex(current_stock_list).sort_values(
            by="score", ascending=False, na_position='last').index

        # 股池pred_score中，去掉已持仓的股票列表，index类型，按score降序
        new = pred_score[~pred_score.index.isin(last)].sort_values(
            by="score", ascending=False).index

        # 取new 的头 topk - (len(last) - n_drop)个股票，index类型，按score降序。这个数量是现有持仓last中，去掉最大卖出数量n_drop，要补齐到topk，需要买入的量。
        min_left = len(
            last) - self.p.n_drop  # n_drop是最大可能卖出支数，卖出n_drop后最小剩余持仓支数minLeft
        max_buy = self.p.topk - min_left  # 最大可能买入支数，使得最终持仓达到topk
        today = new[:max_buy]
        # last和today的并集，index类型，按score降序
        comb = pred_score.reindex(last.union(today)).sort_values(
            by="score", ascending=False, na_position='last').index

        # comb中后n_drop个股票，需要卖出。index类型
        sell = last[last.isin(comb[-self.p.n_drop:])]
        sell_data = [self.getdatabyname(symbol) for symbol in sell]  # 要卖出的行情对象

        # today中头 topk - (len(last) -len(sell))个股票. 买入数量为现有持仓last中，去掉卖出数量len(sell)，要补齐到topk，需要买入的量。index类型
        left = len(last) - len(sell)  # 卖出len(sell)支股票后的剩余持仓支数
        need_buy = self.p.topk - left  # 持仓提升到topk实际需要买入的支数
        buy = today[:need_buy]
        buy_data = [self.getdatabyname(symbol) for symbol in buy]  # 要买入的行情对象
        # # 按持仓市值从大到小排序，以便先卖后买
        # buy_data.sort(key=lambda d: d.close[0] * self.getposition(d).size,
        #               reverse=True)

        # 卖出操作
        for d in sell_data:
            # 卖的数量
            sell_amount = self.getposition(d).size

            ## 注释掉原先的逻辑，生成执行订单的计划
            #o = self.sell(data=d, size=sell_amount)
            #self.order_list.append(o)
            ret_dic[d._name] = (self.SELL_DIRECTION,sell_amount)
            
            trade_value = d.close[0] * sell_amount  # 用今日收盘价估算明日开盘可能的成交金额

            trade_cost = trade_value * (
                self.broker.comminfo[None].p.commission +
                self.broker.comminfo[None].p.stamp_duty)  # 估计交易成本
            cash += (trade_value - trade_cost)  # 估计现金累积值
        # 为要买入的股票每支分配的资金
        to_be_used_cash = cash - total_value*( 1 - self.p.risk_degree)
        cash_per_stock = round(to_be_used_cash / len(buy) if len(buy) > 0 else 0, 2)

        #cash_per_stock = cash * self.p.risk_degree / len(buy) if len(buy) > 0 else 0
        # 买入操作
        for d in buy_data:
            #预先测算待买入的数量
            target_size = math.floor(cash_per_stock / (d.close[0]*self.p.trade_unit))*self.p.trade_unit
            
            if target_size == 0:
                #如果资金允许，至少买入一手，允许其至多45%的资金量（占用闲散现金的1/5)
                if(d.close[0]*self.p.trade_unit <= cash_per_stock*1.45):
                    #print('buy', d._name, ' size ', target_size, ' value', cash_per_stock)
                    ##注释掉原先的逻辑，生成执行订单的计划
                    #o = self.order_target_size(data=d, target=self.p.trade_unit)
                    #self.order_list.append(o)
                    
                    ##买入一手的计划
                    ret_dic[d._name]=(self.BUY_DIRECTION,self.p.trade_unit)
                else:
                    print("目标股票太贵，无法买入：", d._name," 昨日收盘价：",d.close[0])
            elif target_size > 0:
                ##注释掉原先的逻辑，生成执行订单的计划
                #o = self.order_target_value(data=d, target=cash_per_stock)  ## 按价值买入，会用下一个开盘价买入（计入滑点）
                #self.order_list.append(o)
                ##
                ret_dic[d._name]=(self.BUY_DIRECTION,target_size)
            else:
                pass
        print("当前资产：",self.broker.getvalue(),"当前现金：", self.broker.getcash())
        
        return ret_dic

def  read_stock_distribution_file(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 创建high_dict和low_dict
    high_dict = df.set_index('code')[['high_alpha', 'high_beta', 'high_loc', 'high_scale']].apply(tuple, axis=1).to_dict()
    low_dict = df.set_index('code')[['low_alpha', 'low_beta', 'low_loc', 'low_scale']].apply(tuple, axis=1).to_dict()

    # 打印字典以检查结果
    # print("High Distribution Dictionary:")
    # for key, value in high_dict.items():
    #     print(f"{key}: {value}")

    # print("\nLow Distribution Dictionary:")
    # for key, value in low_dict.items():
    #     print(f"{key}: {value}")
    
    return high_dict,low_dict
    
def main(provider_uri=None, provider_day_uri=None, stock_distribution_file=None, exp_name=None, rid=None, pred_score_df=None):
    ##载入qlib数据
    qlib.init(provider_uri=provider_uri, region="cn")
    if pred_score_df is not None:
        pred_df_all = pred_score_df
    else:
        predict_recorder = R.get_recorder(recorder_id=rid,
                                          experiment_name=exp_name)
        pred_df_all = predict_recorder.load_object('pred.pkl')  # 加载预测文件

    start_date = pred_df_all.index[0][0]
    end_date = pred_df_all.index[-1][0]

    stock_pool = list(pred_df_all.index.levels[1])  # 股池，股票列表
    #print("stock_pool:", stock_pool)
    
    ##提取股票的beta分布参数文件
    (high_dist_dic,low_dist_dic) = read_stock_distribution_file(stock_distribution_file)

    
    # 提取ohlc行情数据到df
    df_all = D.features(instruments=stock_pool,
                        fields=[
                            '$open', '$high', '$low', '$close', '$change',
                            '$factor', '$volume'
                        ],
                        start_time=start_date,
                        end_time=end_date,
                        freq="5min")
    # 清洗数据
    df_all = df_all.dropna(subset=["$open", "$high", "$low", "$close"])
    
    ##将ohlc价格修改为除权前的价格
    df_all['$open'] = df_all['$open']/df_all['$factor']
    df_all['$high'] = df_all['$high']/df_all['$factor']
    df_all['$low'] = df_all['$low']/df_all['$factor']
    df_all['$close'] = df_all['$close']/df_all['$factor']
    
    # 创建 stock_pool
    stock_pool = list(pred_df_all.index.levels[1])
    
    #print("df_all:",df_all.head(100))
    #     df_all:                     $open  $high  $low  $close   $change   $factor     $volume
    # instrument datetime                                                                         
    # SH600000   2023-01-03 09:30:00   7.27   7.28  7.22    7.23       NaN  1.386739  1.288161e+06
    #            2023-01-03 09:35:00   7.23   7.25  7.18    7.19 -0.005533  1.386739  1.256192e+06
    #            2023-01-03 09:40:00   7.19   7.20  7.17    7.20  0.001391  1.386739  7.361806e+05
    #            2023-01-03 09:45:00   7.20   7.22  7.18    7.20  0.000000  1.386739  5.536004e+05
    #            2023-01-03 09:50:00   7.20   7.22  7.19    7.20  0.000000  1.386739  5.247875e+05
    #            2023-01-03 09:55:00   7.20   7.22  7.20    7.21  0.001389  1.386739  2.172025e+05
    #            2023-01-03 10:00:00   7.21   7.22  7.20    7.21  0.000000  1.386739  3.718798e+05
    #            2023-01-03 10:05:00   7.22   7.23  7.20    7.21  0.000000  1.386739  2.966673e+05
    #            2023-01-03 10:10:00   7.22   7.22  7.20    7.21  0.000000  1.386739  2.391980e+05
    #            2023-01-03 10:15:00   7.21   7.23  7.20    7.22  0.001387  1.386739  2.401859e+05
    
    #print("df_all:",df_all.tail(20))
    # df_all:                           $open      $high       $low     $close   $change   $factor      $volume
    # instrument datetime                                                                                        
    # SZ300999   2023-01-12 13:20:00  43.940002  43.990002  43.910000  43.919998 -0.000455  0.017906  2647162.500
    #            2023-01-12 13:25:00  43.919998  43.919998  43.850002  43.860001 -0.001366  0.017906  3082771.500
    #            2023-01-12 13:30:00  43.860001  43.930000  43.860001  43.900002  0.000912  0.017906  2434942.750
    #            2023-01-12 13:35:00  43.900002  43.940002  43.889999  43.919998  0.000456  0.017906  2379095.250
    #            2023-01-12 13:40:00  43.919998  43.950001  43.900002  43.919998  0.000000  0.017906  1496707.875
    #            2023-01-12 13:45:00  43.919998  44.030003  43.910000  44.030003  0.002505  0.017906  7818623.000
    #            2023-01-12 13:50:00  44.030003  44.060001  43.990002  43.990002 -0.000908  0.017906  2747687.500
    #            2023-01-12 13:55:00  44.000000  44.020004  43.960003  43.970001 -0.000455  0.017906  1474369.000
    #            2023-01-12 14:00:00  43.980000  44.060001  43.970001  44.030003  0.001365  0.017906  2202282.750
    #            2023-01-12 14:05:00  44.030003  44.080002  44.000000  44.050003  0.000454  0.017906  4975995.000
    #            2023-01-12 14:10:00  44.050003  44.070000  44.030003  44.050003  0.000000  0.017906  1345920.125
    #            2023-01-12 14:15:00  44.050003  44.070000  43.980000  43.980000 -0.001589  0.017906  2312078.500
    #            2023-01-12 14:20:00  43.980000  44.009998  43.940002  43.940002 -0.000910  0.017906  2351171.750
    #            2023-01-12 14:25:00  43.950001  43.970001  43.940002  43.960003  0.000455  0.017906  2296385.500
    #            2023-01-12 14:30:00  43.950001  43.960003  43.930000  43.940002 -0.000455  0.017906  2697425.000
    #            2023-01-12 14:35:00  43.940002  43.940002  43.889999  43.930000 -0.000228  0.017906  2546637.250
    #            2023-01-12 14:40:00  43.930000  43.940002  43.900002  43.900002 -0.000683  0.017906  2848212.750
    #            2023-01-12 14:45:00  43.900002  43.910000  43.860001  43.870003 -0.000683  0.017906  7238481.500
    #            2023-01-12 14:50:00  43.870003  43.910000  43.860001  43.889999  0.000456  0.017906  5310409.000
    #            2023-01-12 14:55:00  43.880001  43.889999  43.860001  43.889999  0.000000  0.017906  7383014.000
            
    #print("df_all:",df_all.sample(10))
    #     df_all:                        $open       $high        $low      $close   $change   $factor       $volume
    # instrument datetime                                                                                             
    # SH601319   2023-01-05 09:50:00    5.360000    5.360000    5.330000    5.340000 -0.003731  0.226693  7.182828e+06
    # SH601728   2023-01-09 13:10:00    4.350000    4.350000    4.340000    4.350000  0.000000  0.176835  5.777148e+06
    # SH603833   2023-01-04 10:40:00  123.510002  124.510002  123.510002  124.250000  0.005666  0.020292  2.089517e+06
    # SH600426   2023-01-12 11:15:00   33.520000   33.610001   33.520000   33.549999  0.000298  0.927098  1.753859e+05
    # SZ001979   2023-01-05 11:00:00   13.589999   13.599999   13.530000   13.530000 -0.005147  0.055245  1.276127e+07
    # SH601377   2023-01-09 14:40:00    6.060000    6.070000    6.050000    6.070000  0.001650  0.177080  1.718090e+07
    # SZ000703   2023-01-12 10:05:00    7.140000    7.140000    7.130001    7.130001 -0.001401  1.637626  5.972059e+04
    # SH601186   2023-01-03 14:35:00    7.890000    7.910000    7.890000    7.900000  0.001267  0.114221  8.216536e+06
    # SZ300750   2023-01-11 11:10:00  428.610016  430.119995  428.610016  429.809998  0.002800  0.027776  6.606414e+06
    # SZ002008   2023-01-09 14:25:00   26.339998   26.349998   26.329998   26.339998  0.000000  0.719838  2.135204e+05

    # 打印缺少数据的股票
    missing_stocks = set(stock_pool) - set(df_all.index.get_level_values(0).unique())
    if missing_stocks:
        print ("Missing stock count:", len(missing_stocks))
        print(f"Warning: The following stocks are missing from df_all: {missing_stocks}")
        
    cerebro = bt.Cerebro(stdstats=False)
    # cerebro.addobserver(bt.observers.Broker)
    # cerebro.addobserver(bt.observers.Trades)
    # cerebro.broker.set_coc(True)  # 以订单创建日的收盘价成交
    # cerebro.broker.set_coo(True)   # 以本日开盘价成交
    starttime = datetime.now()
    
    add_data_num = 0
    skipped_data_num = 0
    for symbol in stock_pool:
        try:
            df = df_all.xs(symbol, level=0)
            data = bt.feeds.PandasDirectData(
                dataname=df,
                datetime=0,  # 日期列为索引
                open=1,  # 开盘价所在列
                high=2,  # 最高价所在列
                low=3,  # 最低价所在列
                close=4,  # 收盘价所在列
                volume=7,  # 成交量所在列
                openinterest=-1,  # 无未平仓量列
                fromdate=start_date,  # 起始日
                todate=end_date,  # 结束日 
                plot=False)
            cerebro.adddata(data, name=symbol)
            add_data_num += 1
        except KeyError:
            skipped_data_num += 1
            #print(f"Error: Could not find data for symbol {e}. Skipping.")
    print("add data:",f"{add_data_num}", "skipped:",f"{skipped_data_num}")
        
    cerebro.addstrategy(TopkDropoutStrategy, pred_df_all=pred_df_all,high_dist=high_dist_dic, low_dist=low_dist_dic)
    startcash = 10000000  ## 一千万初始资金
    cerebro.broker.setcash(startcash)
    # 防止下单时现金不够被拒绝。只在执行时检查现金够不够。
    cerebro.broker.set_checksubmit(False)
    comminfo = StampDutyCommissionScheme()
    cerebro.broker.addcommissioninfo(comminfo)
    # 加入PyFolio分析者
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='PyFolio')
    cerebro.broker.set_slippage_perc(0.0001)  # 百分比滑点
    results = cerebro.run()
    print('最终市值: %.2f' % cerebro.broker.getvalue())

    strat = results[0]  # 获得策略实例

    portfolio_stats = strat.analyzers.getbyname('PyFolio')  # 得到PyFolio分析者实例
    # 以下returns为以日期为索引的资产日收益率系列
    returns, positions, transactions, gross_lev = portfolio_stats.get_pf_items()
    returns.index = returns.index.tz_convert(None)  # 索引的时区要设置一下，否则出错
    
    ##载入qlib day数据
    qlib.init(provider_uri=provider_day_uri, region="cn")
    bench_symbol = 'SH000300'
    df_bench = D.features(
         [bench_symbol],
         fields=['$close'],
         start_time=start_date,
         end_time=end_date,
     ).xs('SH000300').pct_change().rename(columns={'$close': bench_symbol})
    output = "quantstats-tearsheet_bt_1.2_res=0.05.html"
    qs.reports.html(
        returns, benchmark=df_bench, output=output)

    webbrowser.open(output)
    print('耗时',datetime.now()-starttime)


if __name__ == "__main__":
    # ##pred时间段为2023-01-01 至2023-01-15,主要为了测试流程
    #  main(provider_uri=r"/home/godlike/project/GoldSparrow/HighFreq_Data/Qlib_data/hs300_5min_bin",
    #       provider_day_uri=r"/home/godlike/project/GoldSparrow/Updated_Stock_Data",
    #       stock_distribution_file = r"/home/godlike/project/GoldSparrow/Meta_Data/stock_distribution_long_period.csv",
    #       exp_name="LSTM_CSI300_Alpha58",
    #       rid="7c5183bbecbc4ebd95828de1784def47"
    #       )
     
      ##### pred时间段为2023-01-01 至2023-01-30,主要为了测试流程  rid: "0833139cd23a48d592f1a1c6510f8495"
    ##### pred时间段为2023-01-01 至2024-10-30,形成结论  rid: "156de12d5bd8429882e24c11f5593a5b"
    ### pred时间段为2023-01-01 至2024-10-30, ALSTM模型，  rid: 57c61d4d74314018abe86204df221a34

     main(provider_uri=r"/home/godlike/project/GoldSparrow/HighFreq_Data/Qlib_data/hs300_5min_bin",
          provider_day_uri=r"/home/godlike/project/GoldSparrow/Updated_Stock_Data",
          stock_distribution_file = r"/home/godlike/project/GoldSparrow/Meta_Data/stock_distribution_long_period.csv",
          exp_name="LSTM_CSI300_Alpha58",
          rid="0833139cd23a48d592f1a1c6510f8495"
          )