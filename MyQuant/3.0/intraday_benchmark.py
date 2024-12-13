


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
    params = dict(pred_df_all=None, topk=50, n_drop=5, risk_degree=0.9, trade_unit=100)

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
        #if weekday != 5:
        #    return

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

            o = self.sell(data=d, size=sell_amount)
            self.order_list.append(o)
            trade_value = d.close[0] * sell_amount  # 用今日收盘价估算明日开盘可能的成交金额

            trade_cost = trade_value * (
                2 * self.broker.comminfo[None].p.commission +
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
                    o = self.order_target_size(data=d, target=self.p.trade_unit)
                    self.order_list.append(o)
                else:
                    print("目标股票太贵，无法买入：", d._name," 昨日收盘价：",d.close[0])
            elif target_size > 0:
                o = self.order_target_value(data=d, target=cash_per_stock)  ## 按价值买入，会用下一个开盘价买入（计入滑点）
                self.order_list.append(o)
            else:
                pass
        print("当前资产：",self.broker.getvalue(),"当前现金：", self.broker.getcash())

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
    # 清洗数据
    df_all = df_all.dropna(subset=["$open", "$high", "$low", "$close"])
    
    ##将ohlc价格修改为除权前的价格
    df_all['$open'] = df_all['$open']/df_all['$factor']
    df_all['$high'] = df_all['$high']/df_all['$factor']
    df_all['$low'] = df_all['$low']/df_all['$factor']
    df_all['$close'] = df_all['$close']/df_all['$factor']
    
    ###测试涨停板逻辑是否成功，后续要！！注释掉！！！
    #adjustment_date = '2023-01-06'
    #stock_code = 'SH600309'
    #adjustment_factor = 1.15  # 上调 20%
    #price_columns = ['$open', '$high', '$low', '$close']
    #df_all.loc[(stock_code, adjustment_date), price_columns] *= adjustment_factor
    
    # 创建 stock_pool
    stock_pool = list(pred_df_all.index.levels[1])
    print("show some data")
    print(df_all.loc[('SH601966', '2023-01-04')])
    print(df_all.loc[('SZ000568', '2023-01-05')])
    
    #print("df_all:",df_all.sample(20))
    #     df_all:               $open      $high       $low     $close   $change   $factor       $volume
    # instrument datetime                                                                                
    # SZ002353   2023-01-06   3.430601   3.529005   3.421991   3.517934  0.019971  0.123005  7.221659e+05
    # SZ000612   2023-01-04   4.250870   4.266972   4.210616   4.234768 -0.003788  0.805089  7.313228e+04
    # SH601198   2023-01-06   0.660194   0.667763   0.658512   0.661876 -0.001269  0.084101  1.138880e+06
    # SZ000012   2023-01-10   6.105770   6.176974   6.052366   6.150272  0.008759  0.890054  2.580714e+05
    # SH600403   2023-01-03   1.544893   1.558240   1.528209   1.554903  0.010846  0.333670  1.163817e+05
    # SZ002736   2023-01-06   1.269272   1.280592   1.260782   1.267857 -0.001115  0.141502  5.145702e+05
    # SH600585   2023-01-09  18.424379  18.546524  18.231522  18.282951 -0.004899  0.642860  2.177730e+05
    # SZ002275   2023-01-05   0.710832   0.716125   0.703952   0.712420  0.004478  0.052929  5.943786e+05
    # SZ002470   2023-01-03   0.311841   0.319522   0.311841   0.317986  0.000000  0.153616  1.096375e+06
    # SH688009   2023-01-03   0.434028   0.439465   0.433122   0.437653  0.008351  0.090611  1.427610e+06
    # SZ002508   2023-01-10   5.413393   5.510565   5.314454   5.475230  0.015400  0.176677  3.640718e+05
    # SH600028   2023-01-06   3.794988   3.803633   3.751765   3.786344 -0.002278  0.864462  1.240130e+06
    # SH601100   2023-01-03   9.037702   9.125910   8.835258   8.923465 -0.022803  0.144603  3.306533e+05
    # SH601369   2023-01-05   1.342236   1.387506   1.335446   1.382979  0.030354  0.113173  1.366331e+06
    # SH600307   2023-01-06   1.023383   1.029474   1.017291   1.017291 -0.005952  0.609156  2.584380e+05
    # SH600998   2023-01-09   0.779241   0.789061   0.771732   0.779241 -0.002219  0.057764  9.936913e+05
    # SH600267   2023-01-09   2.432291   2.445486   2.408100   2.427892  0.005464  0.219918  4.291740e+05
    # SZ002414   2023-01-05   4.751043   4.793845   4.729642   4.781005  0.005401  0.428022  1.918495e+05
    # SH601633   2023-01-03   9.261097   9.411815   9.155274   9.331646 -0.017556  0.320675  7.638912e+05
    # SZ000758   2023-01-09   4.025000   4.142641   4.025000   4.092223  0.018828  0.840292  3.653103e+05
    #print("df_all:",df_all.head(10))
    #
    # df_all:                  $open      $high       $low     $close   $change   $factor        $volume
    # instrument datetime                                                                                 
    # SH600000   2023-01-03  10.081589  10.095456   9.942915  10.026119 -0.006868  1.386739  186715.234375
    #            2023-01-04  10.081657  10.192596  10.026187  10.137127  0.011065  1.386748  223163.000000
    #            2023-01-05  10.219898  10.233765  10.122829  10.192164  0.005472  1.386689  217512.031250
    #            2023-01-06  10.192039  10.233639  10.136572  10.178172 -0.001361  1.386672  146486.578125
    #            2023-01-09  10.233639  10.233639  10.122705  10.178172  0.000000  1.386672  141434.031250
    #            2023-01-10  10.192218  10.192218  10.095149  10.095149 -0.008174  1.386696  116785.820312
    #            2023-01-11  10.137002  10.178603  10.081532  10.123135  0.002747  1.386731  138567.218750
    #            2023-01-12  10.164357  10.178225  10.039557  10.081157 -0.004110  1.386679  121512.671875
    #            2023-01-13  10.108838  10.192039  10.081104  10.178172  0.009629  1.386672  156996.125000
    #            2023-01-16  10.191913  10.233512  10.094847  10.164179 -0.001362  1.386655  265509.062500
    #
    #print("pred_df_all:",pred_df_all.head(10))
    #
    #pred_df_all:             score
    # datetime   instrument          
    # 2023-01-03 SH600000    0.053400
    #            SH600009    0.084961
    #            SH600010   -0.010217
    #            SH600011   -0.089775
    #            SH600015    0.010321
    #            SH600016    0.000745
    #            SH600018    0.000245
    #            SH600019    0.034792
    #            SH600025   -0.028676
    #            SH600028    0.023601

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
        except KeyError as e:
            print(f"Error: Could not find data for symbol {e}. Skipping.")
        
        
    cerebro.addstrategy(TopkDropoutStrategy, pred_df_all=pred_df_all)
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
    returns, positions, transactions, gross_lev = portfolio_stats.get_pf_items(
    )
    returns.index = returns.index.tz_convert(None)  # 索引的时区要设置一下，否则出错
    bench_symbol = 'SH000300'
    df_bench = D.features(
        [bench_symbol],
        fields=['$close'],
    ).xs('SH000300').pct_change().rename(columns={'$close': bench_symbol})
    output = "quantstats-tearsheet_bt_benchmark.html"
    qs.reports.html(
        returns, benchmark=df_bench, output=output)

    import webbrowser
    webbrowser.open(output)
    print('耗时',datetime.now()-starttime)

if __name__ == "__main__":
    main(provider_uri=r"/home/godlike/project/GoldSparrow/Updated_Stock_Data",
         exp_name="LSTM_CSI300_Alpha58",
         rid="156de12d5bd8429882e24c11f5593a5b" ##"7c5183bbecbc4ebd95828de1784def47"
         )