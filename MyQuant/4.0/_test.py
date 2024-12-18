import baostock as bs
import pandas as pd


def get_valuation_data(code="sh.600000", start_date='2008-01-01', end_date='2008-03-31'):
    """_summary_
获取估值指标：http://baostock.com/baostock/index.php/%E4%BC%B0%E5%80%BC%E6%8C%87%E6%A0%87(%E6%97%A5%E9%A2%91)
    Args:
        code (str): _description_
        start_date (str): _description_
        end_date (str): _description_
    """

    #### 登陆系统 ####
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:'+lg.error_code)
    print('login respond  error_msg:'+lg.error_msg)

    #### 获取沪深A股估值指标(日频)数据 ####
    # peTTM    滚动市盈率
    # psTTM    滚动市销率
    # pcfNcfTTM    滚动市现率
    # pbMRQ    市净率
    rs = bs.query_history_k_data_plus(code,
        "date,open,high,low,close,preclose,volume,amount,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,pbMRQ,isST",
        start_date=start_date, end_date=end_date,
        frequency="d", adjustflag="3")
    print('query_history_k_data_plus respond error_code:'+rs.error_code)
    print('query_history_k_data_plus respond  error_msg:'+rs.error_msg)

    df = rs.get_data()
    print(df.head())
    #### 打印结果集 ####
    result_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        result_list.append(rs.get_row_data())
    result = pd.DataFrame(result_list, columns=rs.fields)

    #### 结果集输出到csv文件 ####
    result.to_csv("./history_A_stock_valuation_indicator_data.csv", encoding="gbk", index=False)
    print(result)

    #### 登出系统 ####
    bs.logout()

def get_daily_data(code="sh.600000", start_date='2015-01-01', end_date='2017-12-31'):
    """_summary_
获取A股K线数据：http://baostock.com/baostock/index.php/A%E8%82%A1K%E7%BA%BF%E6%95%B0%E6%8D%AE
    Args:
        code (str, optional): _description_. Defaults to "sh.600000".
        start_date (str, optional): _description_. Defaults to '2015-01-01'.
        end_date (str, optional): _description_. Defaults to '2017-12-31'.
    """
    #### 登陆系统 ####
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:'+lg.error_code)
    print('login respond  error_msg:'+lg.error_msg)

    #### 获取沪深A股历史K线数据 ####
    # 详细指标参数，参见“历史行情指标参数”章节；“分钟线”参数与“日线”参数不同。“分钟线”不包含指数。
    # 分钟线指标：date,time,code,open,high,low,close,volume,amount,adjustflag
    # 周月线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg
    rs = bs.query_history_k_data_plus(code,
        "date,time,code,open,high,low,close,volume,amount,adjustflag",
        start_date='2024-07-01', end_date='2024-12-31',
        frequency="5", adjustflag="3")
    print('query_history_k_data_plus respond error_code:'+rs.error_code)
    print('query_history_k_data_plus respond  error_msg:'+rs.error_msg)

    #### 打印结果集 ####
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)

    #### 结果集输出到csv文件 ####   
    result.to_csv("D:\\history_A_stock_k_data.csv", index=False)
    print(result)

    #### 登出系统 ####
    bs.logout()
    
get_valuation_data()