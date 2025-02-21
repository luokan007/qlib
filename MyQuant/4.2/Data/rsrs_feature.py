"""
模块文档:
本模块实现了 RSRSFeature 类，用于计算股票价格数据中 RSRS（滚动标准回归斜率）特征。
它支持全量计算和使用缓存机制的增量更新，方便在新增数据时高效地重新计算。
计算的特征包括：
    - base_RSRS: 基础回归斜率，通过 'high' 和 'low' 价格的滑动窗口回归计算得到。
    - norm_RSRS: 标准化 RSRS，通过指定 window_size 的滑动窗口计算移动平均和标准差进行归一化。
    - revise_RSRS: 修正 RSRS，结合普通最小二乘法回归的 R 平方值进行调整。
    - pos_RSRS: revise_RSRS 与 base_RSRS 的乘积。
本模块还提供了测试函数，用于生成合成数据以验证全量计算和增量计算的正确性和性能。

RSRSFeature 类负责针对给定的金融时序数据计算 RSRS 特征。
属性:
    time_range (int): 用于每次滚动回归计算的连续数据点的数量。
    window_size (int): 用于归一化步骤中计算移动平均和标准差的窗口大小。
    cache (dict): 可选的缓存对象，用于存储预计算的 RSRS 特征。
    cache_path (str): 缓存数据的文件路径，用于加载或保存缓存数据。
初始化 RSRSFeature 实例时，需要传入 RSRS 计算所需的参数以及可选的缓存机制。
参数:
    time_range (int): 每次滚动回归计算时考虑的数据点数量。
    window_size (int): 计算滑动平均和标准差时使用的窗口大小。
    cache (dict, 可选): 存储预计算 RSRS 特征的字典对象。
    cache_path (str, 可选): 用于加载或保存缓存的文件路径。如果指定了该路径且 cache 为 None，则尝试从文件加载缓存。
模块支持通过缓存实现增量计算:
    在计算 RSRS 特征时，会检查给定的 stock_id 是否存在缓存中：
        - 如果不存在缓存，则在整个 DataFrame 上进行全量计算。
        - 如果存在缓存，则仅计算自上次缓存时刻之后的新数据的 RSRS 特征。
参数:
    df (pd.DataFrame): 包含至少 'high' 和 'low' 列的价格数据 DataFrame。
    stock_id (str): 股票标识，不能为空。
返回:
    pd.DataFrame: 包含额外 'base_RSRS', 'norm_RSRS', 'revise_RSRS' 和 'pos_RSRS' 列的新 DataFrame。
异常:
    ValueError: 当 stock_id 为空或其他参数不满足要求时抛出该异常。
"""
import time
import pickle

import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm

class RSRSFeature:
    """
    RSRS 特征计算类
    """

    def __init__(self, time_range=18, window_size=252, cache_path=None):
        """
        初始化时设置 time_range 和 window_size，同时可传入缓存对象。
        尝试加载 cache_path 对应的缓存文件（可选）。
        """
        self.time_range = time_range
        self.window_size = window_size
        if cache_path is None:
            self.cache_path = "./.rsrs_feature_cache.pkl"
        else:
            self.cache_path = cache_path
        if Path(self.cache_path).exists():
            # 尝试从文件加载缓存
            try:
                with open(cache_path, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"Cache loaded from {cache_path}")
            except Exception as e:
                print(f"Failed to load cache from {cache_path}: {e}")
                self.cache = {}
        else:            
            self.cache = {}

    def calculate_rsrs_features(self, df: pd.DataFrame, stock_id=None, perf=False):
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
                # df新增四列，分别是base_RSRS, norm_RSRS, revise_RSRS, pos_RSRS，全部填充为空值
                df['base_RSRS'] = np.nan
                df['norm_RSRS'] = np.nan
                df['revise_RSRS'] = np.nan
                df['pos_RSRS'] = np.nan
                return df[['base_RSRS', 'norm_RSRS', 'revise_RSRS', 'pos_RSRS']]
            else:    
                ret_df =  self.full_calculate_rsrs_features(df,perf)
                # 更新缓存
                self._update_cache(ret_df, stock_id)
                return ret_df
        elif status == "incremental":
            
            assert cached_date is not None
            assert cached_df is not None
            assert len(cached_df) >= self.time_range

            # 当传入的df和缓存一致，说明之前已经全部计算过，但由于缓存中仅有base_rsrs特征，因此对df进行全量的计算
            # 这种情况可能是发生在调试时（重复调用），也可能是发生在新增特征后重新计算
            if cached_date == df.index.max():                
                return self.full_calculate_rsrs_features(df, perf)
            
            # 当传入的df日期早于缓存时间，说明是旧数据，为保障兼容性和数据安全性，对df进行全量的计算
            if cached_date > df.index.max():
                return self.full_calculate_rsrs_features(df,perf)
            
            # 有缓存的情况下，使用增量计算的模式
            ret_df = self.incremental_calculate_rsrs_features(df, cached_df, cached_date, perf)
            # 更新缓存
            self._update_cache(ret_df, stock_id)
            return ret_df
        else:
            raise ValueError("未知的缓存状态")

    def incremental_calculate_rsrs_features(self, df: pd.DataFrame, cached_df: pd.DataFrame, last_date, perf=False):
        """
        - 增量计算 RSRS 特征
        - base_RSRS 需要缓存，缓存需要确保有 time_range + n 数据，其中 n 为 df 中 start_idx 之后的数据
        - norm_RSRS 采用滑动窗口的方式计算，窗口大小为 self.window_size
        - revise_RSRS 计算 start_idx 之后的数据，pos_RSRS 无变化
        - 返回 df 中所有列：base_RSRS, norm_RSRS, revise_RSRS, pos_RSRS
        同时增加每个环节的耗时统计，当 perf 为 False 时不运行耗时统计部分
        """
        if last_date == df.index.max():
            raise ValueError("last_date cannot be the maximum index date of the DataFrame.") 
        
        t_total = time.time() if perf else None
        df = df.copy()

        # 1) 计算 base_RSRS 增量部分
        t0 = time.time() if perf else None
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
        if perf:
            t1 = time.time()
            print("incremental base_RSRS 耗时: {:.4f} 秒".format(t1 - t0))
        
        # 2) 计算 norm_RSRS（滑动窗口计算）
        t2 = time.time() if perf else None
        rolling_mean = df['base_RSRS'].rolling(self.window_size).mean()
        rolling_std = df['base_RSRS'].rolling(self.window_size).std()
        df['norm_RSRS'] = (df['base_RSRS'] - rolling_mean) / rolling_std
        if perf:
            t3 = time.time()
            print("incremental norm_RSRS 耗时: {:.4f} 秒".format(t3 - t2))
        
        # 3) 计算 revise_RSRS 增量部分
        t4 = time.time() if perf else None
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
        if perf:
            t5 = time.time()
            print("incremental revise_RSRS 耗时: {:.4f} 秒".format(t5 - t4))
        
        # 4) 计算 pos_RSRS
        t6 = time.time() if perf else None
        df['pos_RSRS'] = df['revise_RSRS'] * df['base_RSRS']
        if perf:
            t7 = time.time()
            print("incremental pos_RSRS 耗时: {:.4f} 秒".format(t7 - t6))
        
        if perf:
            print("incremental total 耗时: {:.4f} 秒".format(t7 - t_total))
        
        return df[['base_RSRS', 'norm_RSRS', 'revise_RSRS', 'pos_RSRS']]

    def full_calculate_rsrs_features(self, df: pd.DataFrame, perf=False):
        """
        全量计算 RSRS 特征，不再传入 time_range 和 window_size，
        使用类初始化时设置的参数 self.time_range 和 self.window_size。
        可选参数 perf 控制是否统计并打印耗时，当 perf 为 False 时不执行时间统计部分。
        """
        df = df.copy()

        # 1) 计算 base_RSRS
        t0 = time.time() if perf else None
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
        if perf:
            print("base_RSRS 时间: {:.4f} 秒".format(time.time() - t0))

        # 2) 计算 norm_RSRS（滑动窗口计算）
        t1 = time.time() if perf else None
        rolling_mean = df['base_RSRS'].rolling(self.window_size).mean()
        rolling_std = df['base_RSRS'].rolling(self.window_size).std()
        df['norm_RSRS'] = (df['base_RSRS'] - rolling_mean) / rolling_std
        if perf:
            print("norm_RSRS 时间: {:.4f} 秒".format(time.time() - t1))

        # 3) 计算 revise_RSRS
        t2 = time.time() if perf else None
        R_square_vals = [np.nan] * self.time_range
        for i in range(len(df) - self.time_range):
            sub_low = df['low'].iloc[i:i + self.time_range]
            sub_high = df['high'].iloc[i:i + self.time_range]
            x_ols = sm.add_constant(sub_low)
            r_val = sm.OLS(sub_high, x_ols).fit().rsquared
            R_square_vals.append(r_val)
        df['revise_RSRS'] = df['norm_RSRS'] * R_square_vals
        if perf:
            print("revise_RSRS 时间: {:.4f} 秒".format(time.time() - t2))

        # 4) 计算 pos_RSRS
        t3 = time.time() if perf else None
        df['pos_RSRS'] = df['revise_RSRS'] * df['base_RSRS']
        if perf:
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
            
    def save_cache(self, cache_path=None):
        """
        将当前的缓存写入到文件中。
        如果传入 cache_path，则使用传入的路径，否则使用初始化时的 cache_path。
        """
        target_path = cache_path if cache_path is not None else self.cache_path
        if target_path is None:
            raise ValueError("cache_path not provided")
        try:
            with open(target_path, 'wb') as f:
                pickle.dump(self.cache, f)
            print(f"Cache saved to {target_path}")
        except Exception as e:
            print(f"Failed to save cache to {target_path}: {e}")

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
    benchmark_df = rsrs_feature_full.full_calculate_rsrs_features(df,perf=True)
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
    cmp_df_1 = rsrs_feature_incremental.calculate_rsrs_features(df_650, stock_id,perf=True)
    end_time_step1 = time.time()
    incremental_times.append(end_time_step1 - start_time_step1)

    # 第二步：增加1行
    df_651 = df_2.iloc[651-window_size-time_range:651]
    start_time_step2 = time.time()
    cmp_df_2 = rsrs_feature_incremental.calculate_rsrs_features(df_651, stock_id,perf=True)
    end_time_step2 = time.time()
    incremental_times.append(end_time_step2 - start_time_step2)

    # 第三步：增加5行
    df_656 = df_2.iloc[656-window_size-time_range:656]
    start_time_step3 = time.time()
    cmp_df_3 = rsrs_feature_incremental.calculate_rsrs_features(df_656, stock_id,perf=True)
    end_time_step3 = time.time()
    incremental_times.append(end_time_step3 - start_time_step3)

    # 第四步：增加44行  
    df_700 = df_2.iloc[700-window_size-time_range:700]
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
# <class 'pandas.core.frame.DataFrame'>
# DatetimeIndex: 700 entries, 2023-01-01 to 2024-11-30
    # Freq: D
    # Data columns (total 4 columns):
    # #   Column       Non-Null Count  Dtype
    # ---  ------       --------------  ----
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

    # 检查数值是否接近
    diff = np.abs(benchmark_df - incremental_df)
    atol = 1e-10
    mask = diff > atol

    difference_info = []
    for col in mask.columns:
        for i, is_diff in enumerate(mask[col]):
            if is_diff:
                difference_info.append({
                    'Row Index': i,
                    'Column': col,
                    'Benchmark Value': benchmark_df.at[i, col],
                    'Incremental Value': incremental_df.at[i, col],
                    'Difference': diff.at[i, col]
                })

    if difference_info:
        difference_df = pd.DataFrame(difference_info)
        print("发现以下差异：")
        print(difference_df)
    else:
        print("未发现任何显著差异。")

def main():
    # 运行测试
    test_rsrs_feature(time_range=30, window_size=500, rows=700)


if __name__ == '__main__':
    main()