# title: get_akshare_data.py
# updated: 2025.1.23
# change log:
#   - 
#   - 

# 目标：
#   1. 从akshare下载基础数据，具体包括：
#       - 两融账户信息，接口: stock_margin_account_info
#       - 中美国债收益率 https://akshare.akfamily.xyz/data/bond/bond.html#id37  接口: bond_zh_us_rate
#       - 300ETF 期权波动率指数，https://akshare.akfamily.xyz/data/index/index.html#id28  接口: index_option_300etf_qvix
#   2. 读取股票列表
#   输入：
#       股票列表文件：{qlib_data}/basic_info.csv   由get_baostock_data.py生成
#   输出：
#       文件夹：Raw_Akshare
#       文件：
#   
