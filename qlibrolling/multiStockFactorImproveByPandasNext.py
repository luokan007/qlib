# 考虑中国佣金，下单量100的整数倍,涨跌停板，滑点
# 考虑一个技术指标，展示怎样处理最小期问题

from datetime import datetime, time
from datetime import timedelta
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




class stampDutyCommissionScheme(bt.CommInfoBase):
    '''
    本佣金模式下，买入股票仅支付佣金，卖出股票支付佣金和印花税.    
    '''
    params = (
        ('stamp_duty', 0.005),  # 印花税率
        ('commission', 0.001),  # 佣金率
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
    )

    def _getcommission(self, size, price, pseudoexec):
        '''
        If size is greater than 0, this indicates a long / buying of shares.
        If size is less than 0, it idicates a short / selling of shares.
        '''

        if size > 0:  # 买入，不考虑印花税
            return size * price * self.p.commission*100
        elif size < 0:  # 卖出，考虑印花税
            return - size * price * (self.p.stamp_duty + self.p.commission*100)
        else:
            return 0  # just in case for some reason the size is 0.



class Strategy(bt.Strategy):
    params = dict(
        rebal_monthday=[1],  # 每月1日执行再平衡
        num_volume=100,  # 成交量取前100名
        period=5,
    )

    # 日志函数
    def log(self, txt, dt=None):
        # 以第一个数据data0，即指数作为时间基准
        dt = dt or self.data0.datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):

        self.lastRanks = []  # 上次交易股票的列表
        # 0号是指数，不进入选股池，从1号往后进入股票池
        self.stocks = self.datas[1:]
        # 记录以往订单，在再平衡日要全部取消未成交的订单
        self.order_list = []
        self.i = -1  # 计数器
        self.interval = 5  # 调仓间隔天数

    def prenext(self):
        self.next()

    def next(self):
        self.i += 1
        # 每隔interval天执行一次调仓
        if self.i % self.interval != 0:
            return

        self.rebalance_portfolio()  # 执行再平衡

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # 订单状态 submitted/accepted，无动作
            return

        # 订单完成
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
            print('毛收益 %0.2f, 扣佣后收益 % 0.2f, 佣金 %.2f, 市值 %.2f, 现金 %.2f' %
                  (trade.pnl, trade.pnlcomm, trade.commission,
                   self.broker.getvalue(), self.broker.getcash()))

    def rebalance_portfolio(self):
        # 从指数取得当前日期
        self.currDate = self.data0.datetime.date(0)
        print('rebalance_portfolio currDate', self.currDate, len(self.stocks))

        # 如果是指数的最后一本bar，则退出，防止取下一日开盘价越界错
        if len(self.datas[0]) == self.data0.buflen():
            return

        # 取消以往所下订单（已成交的不会起作用）
        for o in self.order_list:
            self.cancel(o)
        self.order_list = []  # 重置订单列表

        # for d in self.stocks:
        #     print('sma', d._name, self.sma[d][0],self.sma[d][1], d.marketdays[0])

        # 最终标的选取过程
        # 1 先做排除筛选过程
        self.ranks = [
            d for d in self.stocks if len(d) > 0  # 重要，到今日至少要有一根实际bar
            and d.marketdays > 3 * 365  # 到今天至少上市
            # 今日未停牌 (若去掉此句，则今日停牌的也可能进入，并下订单，次日若复牌，则次日可能成交）（假设原始数据中已删除无交易的记录)
            and d.datetime.date(0) == self.currDate and d.roe >= 0.1 and
            d.pe < 100 and d.pe > 0 and
            len(d) >= self.p.period and d.close[0] > self.sma[d][0]
        ]

        # 2 再做排序挑选过程
        self.ranks.sort(key=lambda d: d.volume, reverse=True)  # 按成交量从大到小排序
        self.ranks = self.ranks[0:self.p.num_volume]  # 取前num_volume名

        if len(self.ranks) == 0:  # 无股票选中，则返回
            return

        # 3 以往买入的标的，本次不在标的中，则先平仓
        # 考虑了跌停的情况。
        # 本日停牌的也允许发单，因为次日有可能成交
        data_toclose = set(self.lastRanks) - set(
            self.ranks)  # data_toclose中可能有今日停牌的股票
        # data_toclose = [d for d in set(self.lastRanks) - set(self.ranks) if d.datetime.date(0) == self.currDate] #  data_toclose中没有今日停牌的股票
        for d in data_toclose:
            lowerprice = d.close[0] * 0.9 + 0.02  # 次日跌停价近似值
            validday = d.datetime.datetime(1)  # 该股下一实际交易日
            self.log(f'平仓 created {d._name}, {self.getposition(d).size}')
            o = self.close(data=d,
                           exectype=bt.Order.Limit,
                           price=lowerprice,
                           valid=validday)
            self.order_list.append(o)  # 记录订单

        # 4 本次标的下单
        # 每只股票买入资金百分比，预留2%的资金以应付佣金和计算误差
        buypercentage = (1 - 0.02) / len(self.ranks)

        # 得到目标市值
        targetvalue = buypercentage * self.broker.getvalue()
        # 为保证先卖后买，股票要按持仓市值从大到小排序
        self.ranks.sort(key=lambda d: self.broker.getvalue([d]), reverse=True)
        self.log('下单, 标的个数 %i, targetvalue %.2f, 当前总市值 %.2f' %
                 (len(self.ranks), targetvalue, self.broker.getvalue()))

        for d in self.ranks:
            # 按次日开盘价计算下单量，下单量是100的整数倍
            size = int(
                abs((self.broker.getvalue([d]) - targetvalue) / d.open[1] //
                    100 * 100))
            validday = d.datetime.datetime(1)  # 该股下一实际交易日
            if self.broker.getvalue([d]) > targetvalue:  # 持仓过多，要卖
                # 次日跌停价近似值
                lowerprice = d.close[0] * 0.9 + 0.02
                self.log(f'sell created {d._name}, {size} ')
                o = self.sell(data=d,
                              size=size,
                              exectype=bt.Order.Limit,
                              price=lowerprice,
                              valid=validday)
            else:  # 持仓过少，要买
                # 次日涨停价近似值

                upperprice = d.close[0] * 1.1 - 0.02
                self.log(f'buy created {d._name}, {size} ')
                o = self.buy(data=d,
                             size=size,
                             exectype=bt.Order.Limit,
                             price=upperprice,
                             valid=validday)

            self.order_list.append(o)  # 记录订单

        self.lastRanks = self.ranks  # 跟踪上次买入的标的


if __name__ == "__main__":
    provider_uri = "G:\qlibrolling\qlib_data\cn_data"  # 原始行情数据存放目录
    qlib.init(provider_uri=provider_uri, region="cn")

    rid = "df620dfb313c4531a804a922167e43e8"
    exp_name = "combine"
    predict_recorder = R.get_recorder(recorder_id=rid,
                                      experiment_name=exp_name)
    pred_df = predict_recorder.load_object('pred.pkl')  # 加载预测文件
    start_date = pred_df.index[0][0]
    end_date = pred_df.index[-1][0]
    stock_pool = list(pred_df.index.levels[1])  # 股池，股票列表
    print('num stocks', len(stock_pool))
    # 提取ohlc行情数据到df
    df_all = D.features(instruments=stock_pool,
                        fields=[
                            '$open', '$high', '$low', '$close', '$change',
                            '$factor', '$volume'
                        ],
                        start_time=start_date,
                        end_time=end_date)
    df_all = df_all[~df_all.isna().any(axis=1)]  # 去掉含nan的行
    print(df_all)

    ##########################
    # 主程序开始
    #########################
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.addobserver(bt.observers.Broker)
    cerebro.addobserver(bt.observers.Trades)
    # cerebro.broker.set_coc(True)  # 以订单创建日的收盘价成交
    # cerebro.broker.set_coo(True) # 以次日开盘价成交

    # datadir = './dataswind'  # 数据文件位于本脚本所在目录的data子目录中
    # datafilelist = glob.glob(os.path.join(datadir, '*'))  # 数据文件路径列表

    # maxstocknum = 20  # 股票池最大股票数目
    # # 注意，排序第一个文件必须是指数数据，作为时间基准
    # datafilelist = datafilelist[0:maxstocknum]  # 截取指定数量的股票池
    # print(datafilelist)
    # # 将目录datadir中的数据文件加载进系统

    for symbol in stock_pool[0:2]:

        df = df_all.xs(symbol)


        data = bt.feeds.PandasData(
            dataname=df,
            datetime=None,  # 日期列为索引
            open=0,  # 开盘价所在列
            high=1,  # 最高价所在列
            low=2,  # 最低价所在列
            close=3,  # 收盘价价所在列
            volume=6,  # 成交量所在列
            openinterest=-1,  # 无未平仓量列
            fromdate=start_date,  # 起始日2002, 4, 1
            todate=end_date,  # 结束日 2015, 12, 31
            plot=False)

        cerebro.adddata(data, name=symbol)

    cerebro.addstrategy(Strategy)
    startcash = 10000000
    cerebro.broker.setcash(startcash)
    # 防止下单时现金不够被拒绝。只在执行时检查现金够不够。
    cerebro.broker.set_checksubmit(False)
    comminfo = stampDutyCommissionScheme(stamp_duty=0.001, commission=0.001)
    cerebro.broker.addcommissioninfo(comminfo)
    results = cerebro.run()
    print('最终市值: %.2f' % cerebro.broker.getvalue())
