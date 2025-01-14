"""_summary_
"""
import os
import pandas as pd
import math
import csv
from pathlib import Path
from typing import Union
from datetime import datetime


class GenerateOrder:
    def __init__(
        self,
        pred_df,
        top_k = 50,
        n_drop = 5,
        working_dir = "/root/autodl-tmp/GoldSparrow/Day_data/Order/",
        position_csv ="position.csv",
        buy_order_csv = "buy.csv",
        sell_order_csv = "sell.csv"
        ):
        self.pred_df = pred_df
        self.working_dir = working_dir
        self.top_k = top_k
        self.n_drop = n_drop
        self.position_csv = position_csv
        self.buy_order_csv = buy_order_csv
        self.sell_order_csv = sell_order_csv
        self.forbid_list = []
        
        self.risk_degree = 0.95
        self.stamp_duty = 0.001 #印花税率
        self.commission = 0.0005 # 佣金率
        self.cash = 57341
         
    def generate_order_csv(self):
        """_summary_

        Args:
            predict_recorder (_type_, optional): _description_. Defaults to None.
        """
            
        working_dir = self.working_dir
        
        #print("pred_df",pred_df)
        
        latest_date = self.pred_df.index.get_level_values('datetime').max()
        
        print("the latest date is:",latest_date)
        
        latest_date_str = latest_date.strftime("%Y_%m_%d")
        order_folder_name = os.path.join(working_dir, latest_date_str)
        if not os.path.exists(order_folder_name):
            os.mkdir(order_folder_name)
        
        pred_score = self.pred_df.xs(latest_date) # 最新一个交易日各股预测分，df
        pred_score = pred_score[
            ~pred_score.index.isin(self.forbid_list)] # 股池去掉禁止持有的股票
        
        ##获得当前持仓
        initial_position_csv_file_path = os.path.join(order_folder_name, self.position_csv)
        if os.path.exists(initial_position_csv_file_path):
            initial_position_df = pd.read_csv(initial_position_csv_file_path, index_col="code")
            initial_position_df.index = initial_position_df.index.str.upper()
            current_stock_list = list(initial_position_df.index)
        else:
            current_stock_list = []
            print("no position file, start with empty position")
            initial_position_df = pd.DataFrame(columns=["code", "amount"])
        
        ##获得当前累积资产
        total_asset = self.cash
        for index, row in initial_position_df.iterrows():
            total_asset += row['quantity'] * row['close']
        print("total asset: ",total_asset)
            
        # 今日已持仓股票列表，按score降序排列。若某持仓股票不在pred_score中，则该股票排在index最后。index类型
        last_tuple = (
            pred_score.reindex(current_stock_list).
            sort_values(by="score", ascending=False, na_position="last").
            index)
        
        # 股池pred_score中，去掉已持仓的股票列表，index类型，按score降序
        new_tuple = (
            pred_score[~pred_score.index.isin(last_tuple)]
            .sort_values(by="score", ascending=False)
            .index)
        
        # 取new 的头 topk - (len(last) - n_drop)个股票，index类型，按score降序。这个数量是现有持仓last中，去掉最大卖出数量n_drop，要补齐到topk，需要买入的量。
        min_left = len(last_tuple) - self.n_drop  # n_drop是最大可能卖出支数，卖出n_drop后最小剩余持仓支数minLeft
        max_buy = self.top_k - min_left  # 最大可能买入支数，使得最终持仓达到topk
        today = new_tuple[:max_buy]
        
        # last和today的并集，index类型，按score降序
        comb = (
            pred_score.reindex(last_tuple.union(today))
            .sort_values(by="score", ascending=False, na_position="last")
            .index)

        # comb中后n_drop个股票，需要卖出。index类型
        sell = last_tuple[last_tuple.isin(comb[-self.n_drop:])]
        # today中头 topk - (len(last) -len(sell))个股票. 买入数量为现有持仓last中，去掉卖出数量len(sell)，要补齐到topk，需要买入的量。index类型
        left = len(last_tuple) - len(sell)  # 卖出len(sell)支股票后的剩余持仓支数
        need_buy = self.top_k - left  # 持仓提升到topk实际需要买入的支数
        buy = today[:need_buy]
        
        sell_order = []
        buy_order = []
        
        # 提取ohlc行情数据到df
        changed_stock = buy.tolist() + sell.tolist()
        #print(changed_stock)
        stock_ohlc_df = D.features(instruments= changed_stock,
                        fields=['$close', '$factor'],
                        start_time=latest_date,
                        end_time=latest_date)
        ##将ohlc价格修改为除权前的价格
        stock_ohlc_df['$close'] = stock_ohlc_df['$close']/stock_ohlc_df['$factor']
        #print(stock_ohlc_df.head())
        
        ##读取股票中文信息，方便后续的可读性
        stock_basic_df = pd.read_csv('/home/godlike/project/GoldSparrow/Meta_Data/stock_basic.csv', index_col='code')
        #print(stock_basic_df.head())
        
        sell_order_file_path = os.path.join(order_folder_name, self.sell_order_csv)
        with open(sell_order_file_path, "w", newline="") as sell_csv_file:
            writer = csv.writer(sell_csv_file)
            # 先写入columns_name
            writer.writerow(["code", "code_name","quantity", "close"])
            
            for symbol in sell:
                # 卖的数量
                sell_amount = initial_position_df.loc[symbol]["quantity"]  # self.getposition(item).size
                sell_order.append((symbol, sell_amount))
                ## 
                close = stock_ohlc_df.at[ (symbol,latest_date), "$close"]
                code_name = stock_basic_df.at[ (symbol), "code_name"]
                writer.writerow([symbol, code_name,sell_amount, close])
                trade_value = close * sell_amount  # 用今日收盘价估算明日开盘可能的成交金额

                trade_cost = trade_value * (self.commission + self.stamp_duty)  # 估计交易成本
                self.cash += trade_value - trade_cost  # 估计现金累积值

        # 为要买入的股票每支分配的资金
        to_be_used_cash = self.cash - total_asset*( 1 - self.risk_degree)
        cash_per_stock = round(to_be_used_cash / len(buy) if len(buy) > 0 else 0, 2)

        #cash_per_stock = self.cash * self.risk_degree / len(buy) if len(buy) > 0 else 0
        
        # 买入操作
        buy_order_file_path = os.path.join(order_folder_name, self.buy_order_csv)
        with open(buy_order_file_path, "w", newline="") as buy_csv_file:
            writer = csv.writer(buy_csv_file)
            # 先写入columns_name
            writer.writerow(["code", "code_name","quantity","close"])
            for symbol in buy:
                close = stock_ohlc_df.at[ (symbol,latest_date), "$close"]
                code_name = stock_basic_df.at[ (symbol), "code_name"]
                 #预先测算待买入的数量
                target_size = math.floor(cash_per_stock / (close * 100))*100
                if target_size == 0:
                    #如果资金允许，至少买入一手，允许其至多45%的资金量（占用闲散现金的1/5)
                    if(close*100 <= cash_per_stock*1.45):
                        target_size = 100
                    else:
                        target_size = 0
                
                writer.writerow([symbol, code_name,target_size,close])
                buy_order.append((symbol, code_name,target_size,close))
                
        # 更新新的持仓
        updated_position_df = initial_position_df.drop(index=sell)
        for symbol, code_name, target_size,close in buy_order:
            assert symbol not in updated_position_df.index
            updated_position_df.loc[symbol] = {'quantity': target_size, 'close': close}  ## 目前数据结构没有价格信息，待后续完善
        
        # 输出新的持仓文件
        new_position_csv_file_path = os.path.join(order_folder_name, f"new_{self.position_csv}")
        updated_position_df.to_csv(new_position_csv_file_path)
        print(f"New position file saved to {new_position_csv_file_path}")
        
def main():
    ## 生成订单文件
    working_dir = "/root/autodl-tmp/GoldSparrow/"

    order_folder_name = os.path.join(working_dir, "Order")
    pred_folder_name = os.path.join(working_dir, "Temp_Data")
    
    Path(order_folder_name).mkdir(parents=True, exist_ok=True)
    
    pkl_path = os.path.join(pred_folder_name, "pred.pkl")
    pred = pd.read_pickle(pkl_path)

    order = GenerateOrder(working_dir=working_dir, pred_df=pred)
    order.generate_order_csv()


if __name__ == "__main__":
    main()