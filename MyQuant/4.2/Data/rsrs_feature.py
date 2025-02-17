import numpy as np
import pandas as pd
import statsmodels.api as sm
import time


class RSRSFeature:
    """
    RSRS 特征计算类
    """

    def __init__(self, time_range=18, window_size=252, cache=None):
        """
        初始化时设置 time_range 和 window_size，同时可传入缓存对象。
        """
        self.time_range = time_range
        self.window_size = window_size
        #中间结果的缓存的数据结构：
        # { key : stock_id
        #  value : (last_date, dataframe)
        #                           index : date
        #                           cols: [base_rsrs, revise_rsrs]}
        # 缓存的设计思路：
        # 计算rsrs特征最耗时的部分在于base_rsrs计算和revise_rsrs需要循环window_size次
        # 优化的方向
        #   1）只计算增量部分的base_rsrs和revise_rsrs，而无需循环整个窗口
        #   2）计算的范围根据传入的df动态调整，无需传入start_idx参数
        # 
        self.cache = cache if cache is not None else {}

    def calculate_rsrs_features(self, df: pd.DataFrame, stock_id=None):
        """
        增量计算 RSRS 特征，并正确加载和更新缓存。
        输入: df，需包含 'high' 和 'low' 列
        返回: 添加了 base_RSRS, norm_RSRS, revise_RSRS, pos_RSRS 列的 DataFrame
        
        第一次调用，使用全量计算的模式
        第二次调用，也就是有缓存的情况下，使用增量计算的模式
        """
        if stock_id is None:
            raise ValueError("stock_id 不能为空")
        # 检查缓存中是否已经有部分结果
        (status, cached_date, cached_df) = self._get_cached_data(stock_id)
        
        ## status = "initial"表示第一次调用，使用全量计算的模式
        ## status = "incremental"表示有缓存的情况下，使用增量计算的模式
        if status == "initial":
            # 第一次调用，使用全量计算的模式
            
            #确保传入的df有time_range的数据（否则无法计算base_rsrs）
            if len(df) < self.time_range:
                #print(f"传入的数据长度不足,计算rsrs特征，需要在{start_date}往前有{self.time_range}个数据")
                # df新增四列，分别是base_RSRS, norm_RSRS, revise_RSRS, pos_RSRS，全部填充为空值
                df['base_RSRS'] = np.nan
                df['norm_RSRS'] = np.nan
                df['revise_RSRS'] = np.nan
                df['pos_RSRS'] = np.nan
                return df[['base_RSRS', 'norm_RSRS', 'revise_RSRS', 'pos_RSRS']]
            else:    
                ret_df =  self.full_calculate_rsrs_features(df)
                # 更新缓存
                self._update_cache(ret_df, stock_id)
                return ret_df
        elif status == "incremental":
            # 有缓存的情况下，使用增量计算的模式
            assert cached_date is not None
            assert cached_df is not None
            assert len(cached_df) >= self.time_range
        
            ret_df = self.incremental_calculate_rsrs_features(df, cached_df, cached_date)
            # 更新缓存
            self._update_cache(ret_df, stock_id)
            return ret_df
        else:
            raise ValueError("未知的缓存状态")
    
    def incremental_calculate_rsrs_features(self, df: pd.DataFrame, cached_df: pd.DataFrame, last_date: pd.DatetimeIndex):
        """
        - 增量计算 RSRS 特征
        - base_RSRS 需要缓存，缓存需要确保有 time_range + n 数据，其中 n 为 df 中 start_idx 之后的数据
        - norm_RSRS 采用滑动窗口的方式计算，窗口大小为 self.window_size
        - revise_RSRS 计算 start_idx 之后的数据，pos_RSRS 无变化
        - 返回 df 中所有列：base_RSRS, norm_RSRS, revise_RSRS, pos_RSRS
        同时增加每个环节的耗时统计
        """
        t_total = time.time()
        df = df.copy()
        
        # 1) 计算 base_RSRS 增量部分
        t0 = time.time()
        if 'base_RSRS' not in df.columns:
            df['base_RSRS'] = np.nan 
        start_idx = df.index.get_loc(last_date) + 1
        loop = len(df) - start_idx
        base_RSRS_vals = []
        for i in range(loop):
            current_idx = start_idx + i
            item = df.iloc[current_idx - self.time_range + 1:current_idx + 1][['high', 'low']]
            if len(item) < self.time_range:
                base_RSRS_vals.append(np.nan)
                continue
            y = item['high']
            x = item['low']
            slope = np.polyfit(x, y, 1)[0]
            base_RSRS_vals.append(slope)
        # 将增量数据赋值，并保留缓存部分数据
        df['base_RSRS'].iloc[start_idx:] = base_RSRS_vals
        df['base_RSRS'].iloc[:start_idx] = cached_df['base_RSRS'].iloc[-start_idx:]
        t1 = time.time()
        print("incremental base_RSRS 耗时: {:.4f} 秒".format(t1 - t0))
        
        # 2) 计算 norm_RSRS（滑动窗口计算）
        t2 = time.time()
        rolling_mean = df['base_RSRS'].rolling(self.window_size).mean()
        rolling_std = df['base_RSRS'].rolling(self.window_size).std()
        df['norm_RSRS'] = (df['base_RSRS'] - rolling_mean) / rolling_std
        t3 = time.time()
        print("incremental norm_RSRS 耗时: {:.4f} 秒".format(t3 - t2))
        
        # 3) 计算 revise_RSRS 增量部分
        t4 = time.time()
        if 'revise_RSRS' not in df.columns:
            df['revise_RSRS'] = np.nan
        R_square_vals = []
        for i in range(loop):
            current_idx = start_idx + i
            sub_low = df['low'].iloc[current_idx - self.time_range:current_idx]
            sub_high = df['high'].iloc[current_idx - self.time_range:current_idx]
            x_ols = sm.add_constant(sub_low)
            r_val = sm.OLS(sub_high, x_ols).fit().rsquared
            R_square_vals.append(r_val)
        df['revise_RSRS'].iloc[start_idx:] = df['norm_RSRS'].iloc[start_idx:] * R_square_vals
        df['revise_RSRS'].iloc[:start_idx] = cached_df['revise_RSRS'].iloc[-start_idx:]
        t5 = time.time()
        print("incremental revise_RSRS 耗时: {:.4f} 秒".format(t5 - t4))
        
        # 4) 计算 pos_RSRS
        t6 = time.time()
        df['pos_RSRS'] = df['revise_RSRS'] * df['base_RSRS']
        t7 = time.time()
        print("incremental pos_RSRS 耗时: {:.4f} 秒".format(t7 - t6))
        
        print("incremental total 耗时: {:.4f} 秒".format(t7 - t_total))
        
        return df[['base_RSRS', 'norm_RSRS', 'revise_RSRS', 'pos_RSRS']]

    def full_calculate_rsrs_features(self, df: pd.DataFrame):
        """
        全量计算 RSRS 特征，不再传入 time_range 和 window_size，
        使用类初始化时设置的参数 self.time_range 和 self.window_size。
        """
        df = df.copy()

        # 1) base_RSRS 耗时测试
        t0 = time.time()
        base_RSRS_vals = []
        for i in range(len(df)):
            start_idx = max(0, i - self.time_range + 1)
            item = df.iloc[start_idx:i + 1][['high', 'low']]
            if len(item) < self.time_range:
                base_RSRS_vals.append(np.nan)
                continue
            y = item['high']
            x = item['low']
            slope = np.polyfit(x, y, 1)[0]
            base_RSRS_vals.append(slope)
        df['base_RSRS'] = base_RSRS_vals
        print("base_RSRS 时间: {:.4f} 秒".format(time.time() - t0))

        # 2) norm_RSRS 耗时测试
        t1 = time.time()
        rolling_mean = df['base_RSRS'].rolling(self.window_size).mean()
        rolling_std = df['base_RSRS'].rolling(self.window_size).std()
        df['norm_RSRS'] = (df['base_RSRS'] - rolling_mean) / rolling_std
        print("norm_RSRS 时间: {:.4f} 秒".format(time.time() - t1))

        # 3) revise_RSRS 耗时测试
        t2 = time.time()
        R_square_vals = [np.nan] * self.time_range
        for i in range(len(df) - self.time_range):
            sub_low = df['low'].iloc[i:i + self.time_range]
            sub_high = df['high'].iloc[i:i + self.time_range]
            x_ols = sm.add_constant(sub_low)
            r_val = sm.OLS(sub_high, x_ols).fit().rsquared
            R_square_vals.append(r_val)
        df['revise_RSRS'] = df['norm_RSRS'] * R_square_vals
        print("revise_RSRS 时间: {:.4f} 秒".format(time.time() - t2))

        # 4) pos_RSRS 耗时测试
        t3 = time.time()
        df['pos_RSRS'] = df['revise_RSRS'] * df['base_RSRS']
        print("pos_RSRS 时间: {:.4f} 秒".format(time.time() - t3))

        return df[['base_RSRS', 'norm_RSRS', 'revise_RSRS', 'pos_RSRS']]

    def _get_cached_data(self, stock_id=None):
        """
        从缓存中获取已有的计算结果。
        """        
        if stock_id in self.cache:
            cached_date, cached_df = self.cache[stock_id]
            status = "incremental"
        else:
            (status, cached_date, cached_df) = ("initial", None, None)

        return status, cached_date, cached_df

    def _update_cache(self, df: pd.DataFrame, stock_id=None):
        """
        将最新的计算结果更新到缓存中，确保使用 df.index 的原始 datetime 类型作为键。
        """
        assert stock_id is not None
        assert len(df) > 0
        if stock_id in self.cache:
            cache_df = self.cache[stock_id][1]
            # 更新缓存数据
            cache_df = cache_df.combine_first(df[["base_RSRS", "revise_RSRS"]])
            self.cache[stock_id] = (cache_df.index.max(), cache_df)
        else:
            cache_df = pd.DataFrame()
            cache_df = df[["base_RSRS", "revise_RSRS"]]
            self.cache[stock_id] = (cache_df.index.max(), cache_df)

def test_rsrs_feature(time_range=10, window_size=100, rows=700):
    """
    测试 RSRS 特征计算的正确性，并统计计算耗时。
    包括全量计算和增量计算方法进行对比。
    """
    stock_id = "sh600001"
    # 2.1 生成虚拟数据
    np.random.seed(42)  # 固定随机种子以保证可重复性
    dates = pd.date_range(start='2023-01-01', periods=rows, freq='D')
    high = np.random.uniform(90, 110, size=rows)
    low = np.random.uniform(80, 100, size=rows)
    df = pd.DataFrame({'high': high, 'low': low}, index=dates)
    df_2 = df.copy()

    # 2.2 使用全量计算方式得到基准结果 benchmark_df，并统计耗时
    rsrs_feature_full = RSRSFeature(time_range=time_range, window_size=window_size)
    start_time_full = time.time()
    benchmark_df = rsrs_feature_full.full_calculate_rsrs_features(df)
    end_time_full = time.time()
    full_calc_time = end_time_full - start_time_full
    print(f"全量计算耗时: {full_calc_time:.4f} 秒")
    print(benchmark_df.tail())

    # 2.3 使用增量计算方式分步计算，并统计每次增量计算耗时及总耗时
    rsrs_feature_incremental = RSRSFeature(time_range=time_range, window_size=window_size)
    incremental_times = []

    # 第一步：先计算前650行数据
    df_650 = df_2.iloc[:650]
    start_time_step1 = time.time()
    cmp_df_1 = rsrs_feature_incremental.calculate_rsrs_features(df_650,stock_id)
    end_time_step1 = time.time()
    incremental_times.append(end_time_step1 - start_time_step1)

    # 第二步：增加1行
    df_651 = df_2.iloc[651-window_size:651]
    start_time_step2 = time.time()
    cmp_df_2 = rsrs_feature_incremental.calculate_rsrs_features(df_651, stock_id)
    end_time_step2 = time.time()
    incremental_times.append(end_time_step2 - start_time_step2)

    # 第三步：增加5行
    df_656 = df_2.iloc[656-window_size:656]
    start_time_step3 = time.time()
    cmp_df_3 = rsrs_feature_incremental.calculate_rsrs_features(df_656, stock_id)
    end_time_step3 = time.time()
    incremental_times.append(end_time_step3 - start_time_step3)

    # 第四步：增加44行  
    df_700 = df_2.iloc[700-window_size:700]
    start_time_step4 = time.time()
    cmp_df_4 = rsrs_feature_incremental.calculate_rsrs_features(df_700, stock_id)
    end_time_step4 = time.time()
    incremental_times.append(end_time_step4 - start_time_step4)

    incremental_df = cmp_df_1.combine_first(cmp_df_2).combine_first(cmp_df_3).combine_first(cmp_df_4)

    total_incremental_time = sum(incremental_times)
    print(f"增量计算总耗时: {total_incremental_time:.4f} 秒")
    print("每次增量计算耗时详情:")
    print(f"  第一步（650行）: {incremental_times[0]:.4f} 秒")
    print(f"  第二步（+1行）: {incremental_times[1]:.4f} 秒")
    print(f"  第三步（+5行）: {incremental_times[2]:.4f} 秒")
    print(f"  第四步（+44行）: {incremental_times[3]:.4f} 秒")

    # 2.4 对比全量计算和增量计算结果
    print("benchmark_df 和 incremental_df 的 info():")
    print(benchmark_df.info())
    print(benchmark_df.tail())
    # <class 'pandas.core.frame.DataFrame'>
    # DatetimeIndex: 700 entries, 2023-01-01 to 2024-11-30
    # Freq: D
    # Data columns (total 4 columns):
    # #   Column       Non-Null Count  Dtype  
    # ---  ------       --------------  -----  
    # 0   base_RSRS    683 non-null    float64
    # 1   norm_RSRS    432 non-null    float64
    # 2   revise_RSRS  432 non-null    float64
    # 3   pos_RSRS     432 non-null    float64
    # dtypes: float64(4)
    # memory usage: 27.3 KB
    
    #             base_RSRS  norm_RSRS  revise_RSRS  pos_RSRS
    # 2024-11-26   0.125307   0.530547     0.005206  0.000652
    # 2024-11-27   0.164302   0.694763     0.011414  0.001875
    # 2024-11-28   0.105165   0.430894     0.014013  0.001474
    # 2024-11-29   0.155548   0.645748     0.009999  0.001555
    # 2024-11-30   0.148283   0.607796     0.020936  0.003104
    
    print(incremental_df.info())
    print(incremental_df.tail())
    # None
    # <class 'pandas.core.frame.DataFrame'>
    # DatetimeIndex: 2657 entries, 2023-01-01 to 2024-11-30
    # Data columns (total 4 columns):
    # #   Column       Non-Null Count  Dtype  
    # ---  ------       --------------  -----  
    # 0   base_RSRS    2589 non-null   float64
    # 1   norm_RSRS    1585 non-null   float64
    # 2   revise_RSRS  1585 non-null   float64
    # 3   pos_RSRS     1585 non-null   float64
    # dtypes: float64(4)
    # memory usage: 103.8 KB
    
    #            base_RSRS  norm_RSRS  revise_RSRS  pos_RSRS
    # 2024-11-26   0.125307   0.530547     0.005206  0.000652
    # 2024-11-27   0.164302   0.694763     0.011414  0.001875
    # 2024-11-28   0.105165   0.430894     0.014013  0.001474
    # 2024-11-29   0.155548   0.645748     0.009999  0.001555
    # 2024-11-30   0.148283   0.607796     0.020936  0.003104

    # 对比全量计算和增量计算结果
    comparison = benchmark_df.compare(incremental_df)
    if (comparison.empty):
        print("测试通过！全量计算和增量计算结果一致。")
    else:
        print("测试失败！全量计算和增量计算结果不一致。")
        print(comparison)
     
    
def main():
    # 运行测试
    test_rsrs_feature(time_range=30, window_size=500,rows=700)


if __name__ == '__main__':
    main()