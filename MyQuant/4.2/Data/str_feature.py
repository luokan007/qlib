import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

class STRFeature:
    def __init__(self, time_range=30, n_jobs=-1):
        self.time_range = time_range
        self.n_jobs = n_jobs
        self._str_factor_df = pd.DataFrame()

    def _calc_str_sigma(self, code_return_df: pd.DataFrame, theta=0.1):
        """_summary_

        Args:
            code_return_df (pd.DataFrame): _description_
            theta (float, optional): _description_. Defaults to 0.1.

        Returns:
            _type_: _description_
        """
        _median = code_return_df.median(axis=1)
        df_frac1 = code_return_df.sub(_median,axis=0).abs()
        df_frac2 = code_return_df.abs().add(_median.abs(),axis=0) + theta
        return df_frac1.div(df_frac2)

    def _calc_str_weight(self, sigma: pd.DataFrame, cur_date, delta=0.7):
        """_summary_
            计算凸显性权重
        Args:
            sigma (_type_): _description_
            time_rage (int, optional): _description_. Defaults to 22.
            delta (float, optional): _description_. Defaults to 0.7.
        """
        # Get all dates up to cur_date and take the last time_range rows
        time_range = self.time_range
        
        df = sigma.loc[:cur_date].iloc[-time_range:]
        df_cleaned = df.dropna(axis=1, how='any')
        df_rank= df_cleaned.rank(axis=0,ascending=False)
        frac1 = df_rank.apply(lambda x: np.power(delta,x))
        frac2 = frac1.mean(axis=1)
        weight = frac1.div(frac2,axis=0)
        assert frac1.iloc[0,1] / frac2.iloc[0] == weight.iloc[0,1]
        return weight

    def _STR_factor(self, weight: pd.DataFrame, return_df: pd.DataFrame, cur_date):
        """_summary_

        Args:
            weight (pd.DataFrame): _description_
        """
        time_range = self.time_range
        ret_df = weight.rolling(time_range).cov(return_df)#.iloc[-1,:]
        #print(ret_df.loc[cur_date])
        #                       code   
        # pctChg  sz002090    0.226865
        #         sz002091   -1.435706
        #         sz002092    0.038047
        #         sz002093    0.268147
        #         sz002094    0.713251
        return ret_df.loc[cur_date]

    def calculate_str_features(self, pivot_df):
        """
            计算每一天的STR因子。
        Args:
            pivot_df (_type_): 每日回报率数据框，其中索引是日期，列为不同的股票代码。
            #          pctChg                                                                                            ...                                                                                                   
# code       sh600300 sh600301 sh600302 sh600303 sh600305 sh600306 sh600307 sh600308 sh600309 sh600310 sh600311  ... sh600888 sh600889 sh600890 sh600891 sh600892 sh600893 sh600894 sh600895 sh600896 sh600897 sh600898
# date                                                                                                           ...                                                                                                   
# 2009-04-15      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN  ...      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN
# 2009-04-16      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN  ...      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN
# 2009-04-17      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN  ...      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN
# 2009-04-20      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN  ...      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN
# 2009-04-21      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN  ...      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN
# ...             ...      ...      ...      ...      ...      ...      ...      ...      ...      ...      ...  ...      ...      ...      ...      ...      ...      ...      ...      ...      ...      ...      ...
# 2025-01-27  10.0334  -0.9556   0.0000  -0.3597   0.5355      NaN   0.0000   1.1730   0.8091  -0.4902      NaN  ...   0.1427  -9.8867      NaN      NaN   3.6424   0.0268   2.4162  -3.4068      NaN   0.8253      NaN
# 2025-02-05   0.3040   0.9081   3.5941  -0.3610  -0.1332      NaN   2.0408  -0.8696  -1.1674  -1.7241      NaN  ...  -0.2849   1.3143      NaN      NaN  -1.9169   1.6069  -5.4795   2.3651      NaN  -1.2278      NaN
# 2025-02-06   0.3030   2.6434   0.4082   0.7246   0.1333      NaN   0.6667   4.0936  -0.9006   2.0050      NaN  ...   0.5714   9.9831      NaN      NaN   2.2801   2.8730  -1.0467   4.0130      NaN   0.1381      NaN
# 2025-02-07   1.2085  -0.8219   0.0000   3.9568   0.5326      NaN   1.3245   0.2809   2.5179   0.9828      NaN  ...   0.7102   1.7949      NaN      NaN   3.1847  -0.5124   0.0814   1.8706      NaN  -0.0690      NaN
        Returns:
            pd.DataFrame: : 每天的STR因子数据框，其中索引是日期，列为不同的股票代码。
#                       pctChg                                                                                                                                           
# code        sz002090  sz002091  sz002092  sz002093  sz002094  sz002095  sz002096 sz002097  sz002098  sz002099  sz002100  sz002101  sz002102  sz002103  sz002104
# date                                                                                                                                                           
# 2008-01-02       NaN       NaN       NaN       NaN       NaN       NaN       NaN      NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN
# 2008-01-03       NaN       NaN       NaN       NaN       NaN       NaN       NaN      NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN
# 2008-01-04       NaN       NaN       NaN       NaN       NaN       NaN       NaN      NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN
# 2008-01-07       NaN       NaN       NaN       NaN       NaN       NaN       NaN      NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN
# 2008-01-08       NaN       NaN       NaN       NaN       NaN       NaN       NaN      NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN
# 2008-01-09       NaN       NaN       NaN       NaN       NaN       NaN       NaN      NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN
# 2008-01-10  0.226865 -1.435706  0.038047  0.268147  0.713251 -0.198438   1.18973      NaN  0.647639 -0.375385 -0.418327  0.186402 -0.698619  1.999057  0.424592
# 2008-01-11 -0.134092 -0.225108 -0.188916 -0.137051 -0.129813  -0.75082  0.756377      NaN    0.6385 -0.676059       NaN -0.496476 -0.597598    -0.235 -0.339108    
        """
        ## 检查pivot_df每一行的数据，如果空值大于5条，则drop掉该行，并打印drop掉的行的索引
        before_drop = len(pivot_df)
        pivot_df = pivot_df.dropna(thresh=5)
        print("drop NA line for STR computing:", before_drop - len(pivot_df))

        time_range = self.time_range
        # 创建一个空的DataFrame来存储每天的STR因子
        str_factors = pd.DataFrame(index=pivot_df.index, columns=pivot_df.columns)

        sigma = self._calc_str_sigma(pivot_df)
        
        def _calculate_str_factor_for_date(cur_date, sigma, return_df):
            # 为每一天计算weight

            weight = self._calc_str_weight(sigma=sigma, cur_date=cur_date)
            str_factor = self._STR_factor(weight=weight, return_df=return_df, cur_date=cur_date)
            return str_factor

        # 遍历每一天
        dates_to_process = pivot_df.index[time_range:]
        results = []
        
        # for date in tqdm(dates_to_process, desc='Calculating STR factors'):
        #     result = _calculate_str_factor_for_date(date, sigma, pivot_df)
        #     results.append(result)
        
        # 使用joblib进行并行计算
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_calculate_str_factor_for_date)(date, sigma, pivot_df)
            for date in tqdm(dates_to_process, desc='Calculating STR factors')
        )
        
        # 将结果存入str_factors DataFrame
        for date, str_factor in zip(dates_to_process, results):
            str_factors.loc[date] = str_factor

        # # 遍历每一天
        # for date in pivot_df.index[time_range:]:

        #     # 为每一天计算weight
        #     weight = self._calc_str_weight(sigma=sigma, cur_date=date, time_range=time_range)
        #     str_factor = self._STR_factor(weight=weight, return_df=pivot_df, cur_date=date, time_range=time_range)

        #     # 将STR因子存入str_factors DataFrame
        #     str_factors.loc[date] = str_factor
            
        #     #print(str_factors.head(20))

        return str_factors
