# title: rsrs_feature.py
# updated: 2025.1.23
# change log:
#   - 实现RSRS因子计算
#   

# 目标：
#

import pandas as pd
import numpy as np
import statsmodels.api as sm

class RSRSFeature:
    """_summary_
    """
    
    @staticmethod
    def calculate_rsrs_features(df: pd.DataFrame, time_range=18, window_size=252):
        """
        将 _cal_base_RSRS, _cal_norm_RSRS, _cal_revise_RSRS, _cal_pos_RSRS 合并到一个函数中
        输入: df，需包含 'high', 'low' 列
        返回: 添加了 base_RSRS, norm_RSRS, revise_RSRS, pos_RSRS 列的 DataFrame
        """
        df = df.copy()

        # 1) base_RSRS (使用滚动+polyfit计算RSRS)
        base_RSRS_vals = []
        for item in df[['high','low']].rolling(time_range):
            if len(item) < time_range:
                base_RSRS_vals.append(np.nan)
                continue
            y = item['high']
            x = item['low']
            base_RSRS_vals.append(np.polyfit(x, y, 1)[0])
        df['base_RSRS'] = base_RSRS_vals
        
        # 2) norm_RSRS
        df['norm_RSRS'] = (df['base_RSRS'] - df['base_RSRS'].rolling(window_size).mean()) \
                          / df['base_RSRS'].rolling(window_size).std()

        # 3) revise_RSRS
        R_square_vals = [np.nan]*time_range
        for i in range(len(df)-time_range):
            sub_low = df['low'].iloc[i:i+time_range]
            sub_high = df['high'].iloc[i:i+time_range]
            x_ols = sm.add_constant(sub_low)
            r_val = sm.OLS(sub_high, x_ols).fit().rsquared
            R_square_vals.append(r_val)
        df['revise_RSRS'] = df['norm_RSRS'] * R_square_vals

        # 4) pos_RSRS
        df['pos_RSRS'] = df['revise_RSRS'] * df['base_RSRS']
        
        return df[['base_RSRS','norm_RSRS','revise_RSRS','pos_RSRS']]

    @staticmethod
    def __test__():
        """
        测试用例演示
        """
        data = {
            'date': pd.date_range(start='2023-01-01', periods=300, freq='D'),
            'high': np.random.uniform(10, 100, 300),
            'low': np.random.uniform(5, 50, 300),
        }
        df_test = pd.DataFrame(data).set_index('datetime')
        result = RSRSFeature.calculate_rsrs_features(df_test, 18, 252)
        print(result.tail(10))

if __name__ == "__main__":
    RSRSFeature.__test__()

