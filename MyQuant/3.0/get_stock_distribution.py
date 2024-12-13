from scipy import stats
import math
import pandas as pd
import numpy as np
import csv
import qlib

from qlib.data import D  # 基础行情数据服务的对象




class GetStockDistribution:
    """_summary_
    """
    def __init__(self,provider_day_uri=None,stock_basic_csv=None):
        if provider_day_uri is not None:
            self.provider_day_uri = provider_day_uri
        else:
            self.provider_day_uri = r"/home/godlike/project/GoldSparrow/Updated_Stock_Data"
        if stock_basic_csv is not None:
            self.stock_basic_csv = stock_basic_csv
        else:
            self.stock_basic_csv = r"/home/godlike/project/GoldSparrow/Meta_Data/stock_basic.csv"
        self.stock_pool = 'csi300'
    def get_stock_distribution(self, start_date, end_date, output_distribution_file_name=None):
        qlib.init(provider_uri=self.provider_day_uri, region="cn")
        ##获取CSI300的股票列表
        stock_list = D.instruments(market=self.stock_pool)
        print(stock_list)
        
        ##获取股票行情信息
        df_all = D.features(instruments=stock_list,
                        fields=['$open', '$high', '$low', '$close'],
                        start_time=start_date,
                        end_time=end_date)
        # 清洗数据
        df_all = df_all.dropna(subset=["$open", "$high", "$low", "$close"])
        
        ##计算正值和负值,有涨跌幅10%限制，映射到0-1区间
        df_all['$high_pos'] = (df_all['$high']/df_all['$open'] - 1)/0.1
        df_all['$low_pos'] = (1 - df_all['$low']/df_all['$open'])/0.1
        
        upper_bound = 1 - np.exp(-7)
        lower_bound = np.exp(-7)
        
        df_all.loc[df_all['$high_pos'] >= 1, '$high_pos'] = upper_bound
        df_all.loc[df_all['$low_pos'] >= 1, '$low_pos'] = upper_bound
        df_all.loc[df_all['$high_pos'] <=0, '$high_pos'] = lower_bound
        df_all.loc[df_all['$low_pos'] <=0, '$low_pos'] = lower_bound
        
        ##将high_pos和low_pos分别导出为list
        # 初始化字典来存储每个symbol对应的high_pos和low_pos列表
        symbol_high_pos = {}
        symbol_low_pos = {}

        # 使用groupby方法按照'instrument'分组
        for symbol, group in df_all.groupby(level='instrument'):
            # 将每个symbol的'high_pos'和'low_pos'列转换成列表并存入相应的字典
            high_pos_list = group['$high_pos'].tolist()
            low_pos_list = group['$low_pos'].tolist()
            
            ##将样本点输出到文件
            if symbol in ['SH601198','SH600654']:
                with open(f'{symbol}.csv', mode='w') as file:
                    writer = csv.writer(file)
                    writer.writerow(['high_pos','low_pos'])
                    if len(high_pos_list) == len(low_pos_list):
                        for i in range(len(high_pos_list)):
                            writer.writerow([high_pos_list[i],low_pos_list[i]])
   
            #使用beta分布来拟合
            try:
                high_alpha, high_beta,high_loc, high_scale = stats.beta.fit(high_pos_list,floc=0, fscale=1)
                low_alpha, low_beta,low_loc, low_scale = stats.beta.fit(low_pos_list,floc=0, fscale=1)
                symbol_high_pos[symbol]=(high_alpha, high_beta,high_loc, high_scale)
                symbol_low_pos[symbol]=(low_alpha, low_beta,low_loc, low_scale)
            except:
                print(f"{symbol} has no sufficient data")
                symbol_high_pos[symbol]=(0.5, 3,0, 1)
                symbol_low_pos[symbol]=(0.5, 3,0, 1)
                
            
        with open(output_distribution_file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['code', 'high_alpha', 'high_beta', 'high_loc', 'high_scale',
                            'low_alpha', 'low_beta', 'low_loc', 'low_scale'])
            for symbol, (high_alpha, high_beta,high_loc, high_scale) in symbol_high_pos.items():
                (low_alpha, low_beta,low_loc, low_scale) = symbol_low_pos[symbol]
                # 格式化浮点数为小数点后六位
                output_data = [symbol,
                            f"{high_alpha:.6f}", f"{high_beta:.6f}", f"{high_loc:.6f}", f"{high_scale:.6f}",
                            f"{low_alpha:.6f}", f"{low_beta:.6f}", f"{low_loc:.6f}", f"{low_scale:.6f}"]
                writer.writerow(output_data)
       
            # 使用最大似然估计法拟合正态分布
            ##拟合正态分布，分别拟合正值( high/open - 1),以及负值(low/open - 1)的分布
        #     if len(high_pos_list) >0:
        #         high_mu_mle, high_sigma_mle = stats.norm.fit(high_pos_list)
        #         symbol_high_pos[symbol] = (high_mu_mle, high_sigma_mle)
        #     else:
        #         symbol_high_pos[symbol] = (0,0)
            
        #     if len(low_pos_list) >0:
        #         low_mu_mle, low_sigma_mle = stats.norm.fit(low_pos_list)
        #         symbol_low_pos[symbol] = (low_mu_mle, low_sigma_mle)
        #     else:
        #         symbol_low_pos[symbol] = (0,0)
                
                
        # if output_distribution_file_name is not None:
        #     ##输出数值到文件中，格式为：code,high_distribution,low_distribution
        #     # 将结果写入CSV文件
        #     with open(output_distribution_file_name, mode='w', newline='') as file:
        #         writer = csv.writer(file)
        #         writer.writerow(['code', 'high_mean', 'high_std', 'low_mean','low_std'])  # 写入表头
        #         for symbol, (high_mu_mle, high_sigma_mle) in symbol_high_pos.items():
        #             (low_mu_mle, low_sigma_mle) = symbol_low_pos[symbol]
        #             # 格式化浮点数为小数点后六位
        #             output_data = [symbol,
        #                         f"{high_mu_mle:.6f}", f"{high_sigma_mle:.6f}", 
        #                         f"{low_mu_mle:.6f}", f"{low_sigma_mle:.6f}"]
        #             writer.writerow(output_data)


        return (symbol_high_pos,symbol_low_pos)

if __name__ == "__main__":
    gsd = GetStockDistribution(provider_day_uri= r"/home/godlike/project/GoldSparrow/Updated_Stock_Data",
                               stock_basic_csv = r"/home/godlike/project/GoldSparrow/Meta_Data/stock_basic.csv")
    
    (long_symbol_high,long_symbol_low) = gsd.get_stock_distribution(start_date="2008-01-01", end_date="2024-10-31",
                              output_distribution_file_name=r"/home/godlike/project/GoldSparrow/Meta_Data/stock_distribution_long_period.csv")
    
    (short_symbol_high,short_symbol_low) = gsd.get_stock_distribution(start_date="2024-01-01", end_date="2024-12-05",
                              output_distribution_file_name=r"/home/godlike/project/GoldSparrow/Meta_Data/stock_distribution_short_term.csv")
    