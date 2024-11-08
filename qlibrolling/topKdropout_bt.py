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
import quantstats as qs


class stampDutyCommissionScheme(bt.CommInfoBase):
    '''
    本佣金模式下，买入股票仅支付佣金，卖出股票支付佣金和印花税.    
    '''
    params = (
        ('stamp_duty', 0.001),  # 印花税率
        ('commission', 0.0005),  # 佣金率
        ('percabs', True),
        ('stocklike', True),
    )

    def _getcommission(self, size, price, pseudoexec):
        '''
        If size is greater than 0, this indicates a long / buying of shares.
        If size is less than 0, it idicates a short / selling of shares.
        '''

        if size > 0:  # 买入，不考虑印花税
            return size * price * self.p.commission
        elif size < 0:  # 卖出，考虑印花税
            return -size * price * (self.p.stamp_duty + self.p.commission)
        else:
            return 0  # just in case for some reason the size is 0.




class TopkDropoutStrategy(bt.Strategy):
    params = dict(pred_df_all=None, topk=50, n_drop=5, risk_degree=0.95)

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

    def prenext(self):
        self.next()

    def next(self):

        for o in self.order_list:
            self.cancel(o)  # 取消所有未执行订
        self.order_list = []  # 重置
        weekday = self.datetime.date(0).isoweekday()  # 今天周几

        # 每周5晚上调仓
        if weekday != 5:
            return

        p_list = [p for p in self.getpositions().values() if p.size != 0]
        print('num position', len(p_list))  # 持仓品种数

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
                    0) < self.end_date - timedelta(days=2 *
                                                   self.notify_delistdays):
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

        if len(buy) != 0:
            print('buy order', buy)
        # 卖出操作
        for d in sell_data:
            # 卖的数量
            sell_amount = self.getposition(d).size

            o = self.sell(data=d, size=sell_amount)
            self.order_list.append(o)
            trade_value = d.close[0] * sell_amount  # 用今日收盘价估算明日开盘可能的成交金额

            trade_cost = trade_value * (
                2 * self.broker.comminfo[None].p.commission +
                self.broker.comminfo[None].p.stamp_duty)  # 估计交易成本
            cash += (trade_value - trade_cost)  # 估计现金累积值
        # 为要买入的股票每支分配的资金
        cash_per_stock = cash * self.p.risk_degree / len(buy) if len(
            buy) > 0 else 0
        # 买入操作
        for d in buy_data:
            print('buy', d._name, 'value', cash_per_stock)
            o = self.order_target_value(data=d, target=cash_per_stock)
            self.order_list.append(o)


def main(provider_uri=None, exp_name=None, rid=None, pred_score_df=None):
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
    # 提取ohlc行情数据到df
    df_all = D.features(instruments=stock_pool,
                        fields=[
                            '$open', '$high', '$low', '$close', '$change',
                            '$factor', '$volume'
                        ],
                        start_time=start_date,
                        end_time=end_date)
    
    df_all = df_all.dropna(subset=["$open","$high","$low","$close"])  # 去掉含nan的行


    cerebro = bt.Cerebro(stdstats=False)
    # cerebro.addobserver(bt.observers.Broker)
    # cerebro.addobserver(bt.observers.Trades)
    # cerebro.broker.set_coc(True)  # 以订单创建日的收盘价成交
    # cerebro.broker.set_coo(True)   # 以本日开盘价成交
    starttime = datetime.now()
    for symbol in stock_pool:

        df = df_all.xs(symbol)
        data = bt.feeds.PandasDirectData(
            dataname=df,
            datetime=0,  # 日期列为索引
            open=1,  # 开盘价所在列
            high=2,  # 最高价所在列
            low=3,  # 最低价所在列
            close=4,  # 收盘价价所在列
            volume=7,  # 成交量所在列
            openinterest=-1,  # 无未平仓量列
            fromdate=start_date,  # 起始日
            todate=end_date,  # 结束日 
            plot=False)

        cerebro.adddata(data, name=symbol)

    
    cerebro.addstrategy(TopkDropoutStrategy, pred_df_all=pred_df_all)
    startcash = 100000000
    cerebro.broker.setcash(startcash)
    # 防止下单时现金不够被拒绝。只在执行时检查现金够不够。
    cerebro.broker.set_checksubmit(False)
    comminfo = stampDutyCommissionScheme(stamp_duty=0.001, commission=0.0005)
    cerebro.broker.addcommissioninfo(comminfo)
    # 加入PyFolio分析者
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='PyFolio')
    cerebro.broker.set_slippage_perc(0.001)  # 百分比滑点
    results = cerebro.run()
    print('最终市值: %.2f' % cerebro.broker.getvalue())

    strat = results[0]  # 获得策略实例

    portfolio_stats = strat.analyzers.getbyname('PyFolio')  # 得到PyFolio分析者实例
    # 以下returns为以日期为索引的资产日收益率系列
    returns, positions, transactions, gross_lev = portfolio_stats.get_pf_items(
    )
    returns.index = returns.index.tz_convert(None)  # 索引的时区要设置一下，否则出错
    bench_symbol = 'SH000300'
    df_bench = D.features(
        [bench_symbol],
        fields=['$close'],
    ).xs('SH000300').pct_change().rename(columns={'$close': bench_symbol})
    output = "quantstats-tearsheet_bt.html"
    qs.reports.html(
        returns, benchmark=df_bench, output=output)

    import webbrowser
    webbrowser.open(output)
    print('耗时',datetime.now()-starttime)





if __name__ == "__main__":

    main(provider_uri=r"G:\qlibrolling\qlib_data\cn_data_rolling",
         exp_name="combine",
         rid="d8253fe24a584a93aeb72ba743868700" # "df620dfb313c4531a804a922167e43e8" 
         )
