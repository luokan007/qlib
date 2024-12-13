#### 查询Baostock，获取证券基本资料


import baostock as bs
import pandas as pd

# 登陆系统
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)

# 获取证券基本资料
rs = bs.query_stock_basic()
#print(rs)
# rs = bs.query_stock_basic(code_name="浦发银行")  # 支持模糊查询
print('query_stock_basic respond error_code:'+rs.error_code)
print('query_stock_basic respond  error_msg:'+rs.error_msg)

# 打印结果集
data_list = []
while (rs.error_code == '0') & rs.next():
    # 获取一条记录，将记录合并在一起
    ## 将名字格式化为：SH600000
    ## 仅保留type=1(股票),status=1(上市中)的数据
    tmp = rs.get_row_data()
    
    if(tmp[4] == '1' and tmp[5] == '1'):
        #print(tmp)
        tmp[0] = tmp[0].upper().replace('.', '')  
        data_list.append(tmp)
result = pd.DataFrame(data_list, columns=rs.fields)
# 设置 'code' 为索引并检查唯一性
result.set_index('code', inplace=True, verify_integrity=True)
#print(result)

# 结果集输出到csv文件
result.to_csv("/home/godlike/project/GoldSparrow/Meta_Data/stock_basic.csv", encoding="utf8", index=True)


# 登出系统
bs.logout()