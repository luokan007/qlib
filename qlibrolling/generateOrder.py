from qlib.workflow import R
import qlib
import pandas as pd

# 初始仓位和现金设置
cash = 100000  # 初始资金
initia_position_df = pd.read_csv("initialPosition.csv", index_col="code")

initia_position_df.index = initia_position_df.index.str.upper()
# print(initia_position_df)

# TopkDropout 参数
n_drop = 5
topk = 50
commission = 0.0005  # 佣金率
stamp_duty = 0.001  # 印花税率
risk_degree = 0.95


# 获取预测结果
qlib.init()

experiment_id = "28"
recorder_id = "23273457919441c286f6682b40618c5b"

predict_recorder = R.get_recorder(experiment_id=experiment_id, recorder_id=recorder_id)
pred_df = predict_recorder.load_object("pred.pkl")
# print("pred_df")
# print(pred_df.sort_values("score"))

forbid = [] # 禁止持有的股票
curr_date = pred_df.index[0][0]
print("curr_date", curr_date)

pred_score = pred_df.xs(curr_date)  # 本日各股预测分，df
pred_score = pred_score[
            ~pred_score.index.isin(forbid)]  # 股池去掉禁止持有的股票
# print(pred_score)

current_stock_list = list(initia_position_df.index)

# 今日已持仓股票列表，按score降序排列。若某持仓股票不在pred_score中，则该股票排在index最后。index类型
last = (
    pred_score.reindex(current_stock_list)
    .sort_values(by="score", ascending=False, na_position="last")
    .index
)

# 股池pred_score中，去掉已持仓的股票列表，index类型，按score降序
new = (
    pred_score[~pred_score.index.isin(last)]
    .sort_values(by="score", ascending=False)
    .index
)

# 取new 的头 topk - (len(last) - n_drop)个股票，index类型，按score降序。这个数量是现有持仓last中，去掉最大卖出数量n_drop，要补齐到topk，需要买入的量。
min_left = len(last) - n_drop  # n_drop是最大可能卖出支数，卖出n_drop后最小剩余持仓支数minLeft
max_buy = topk - min_left  # 最大可能买入支数，使得最终持仓达到topk
today = new[:max_buy]
# last和today的并集，index类型，按score降序
comb = (
    pred_score.reindex(last.union(today))
    .sort_values(by="score", ascending=False, na_position="last")
    .index
)

# comb中后n_drop个股票，需要卖出。index类型
sell = last[last.isin(comb[-n_drop:])]
# today中头 topk - (len(last) -len(sell))个股票. 买入数量为现有持仓last中，去掉卖出数量len(sell)，要补齐到topk，需要买入的量。index类型
left = len(last) - len(sell)  # 卖出len(sell)支股票后的剩余持仓支数
need_buy = topk - left  # 持仓提升到topk实际需要买入的支数
buy = today[:need_buy]

sell_order = []
buy_order = []


# 卖出操作
import csv
with open("sell_order.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # 先写入columns_name
    writer.writerow(["code", "quantity", "close"])

    for d in sell:
        # 卖的数量
        sell_amount = initia_position_df.loc[d]["quantity"]  # self.getposition(d).size
        sell_order.append((d, sell_amount))
        close = initia_position_df.loc[d]["close"]
        writer.writerow([d, sell_amount, close])
        trade_value = close * sell_amount  # 用今日收盘价估算明日开盘可能的成交金额

        trade_cost = trade_value * (2 * commission + stamp_duty)  # 估计交易成本
        cash += trade_value - trade_cost  # 估计现金累积值


# 为要买入的股票每支分配的资金
cash_per_stock = round(cash * risk_degree / len(buy) if len(buy) > 0 else 0, 2)


# 买入操作
with open("buy_order.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # 先写入columns_name
    writer.writerow(["code", "value"])
    for d in buy:
        # print("buy", d, "value", cash_per_stock)
        writer.writerow([d, cash_per_stock])
        buy_order.append((d, cash_per_stock))


print("sell_order", sell_order)
print("buy_order", buy_order)
