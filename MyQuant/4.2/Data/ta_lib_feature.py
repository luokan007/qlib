"""_summary_

    Returns:
        _type_: _description_
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import talib
from joblib import Parallel, delayed
from tqdm import tqdm  
from rsrs_feature import RSRSFeature



class TALibFeatureExt:
    """_summary_
    """
    def __init__(self, basic_info_path,time_range = 30, stock_pool_path = None):
        """初始化特征生成器"""
        # 定义要生成的技术指标及其参数
        
        self.time_range = time_range
        
        self.feature_functions = {
            'EMA': {'timeperiod': [5, 10, 20]},
            'SAR': {},
            'KAMA': {'timeperiod': [12, 24, 48]},
            'TEMA': {'timeperiod': [12, 24, 48]},
            'TRIMA': {'timeperiod': [12, 24, 48]},

            'ADX': {'timeperiod': [14,28]},
            'APO ': {},
            'AROON': {'timeperiod': [14,28]},
            'BOP': {},
            'CCI': {'timeperiod': [14,28]},
            'CMO': {'timeperiod': [14,28]},
            'MACD': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9},
            'MACDEXT': {},
            'MOM': {'timeperiod': [6, 12, 24, 48]},
            'MFI': {'timeperiod': [6, 12, 24, 48]},
            'ROC': {'timeperiod': [6, 12, 24, 48]},
            'RSI': {'timeperiod': [6, 12, 24]},
            'STOCHF': {},
            'STOCHRSI': {},
            'TRIX': {'timeperiod': [12, 24, 48]},
            'ULTOSC': {},
            'WILLR': {'timeperiod': [6, 12, 24, 48]},

            'AD': {},
            'ADOSC': {'fastperiod': 3, 'slowperiod': 10},
            'OBV': {},

            'ATR': {'timeperiod': [14, 28]},
            'NATR': {'timeperiod': [14, 28]},
            'TRANGE': {},
            
            'LINEARREG_SLOPE': {'timeperiod': [5, 14, 28]},
            'TSF': {'timeperiod': [5, 10, 20, 40]},
            'VAR': {'timeperiod': [5, 10, 20, 40]},

            'TURN_RATE_LN': {},
            'TURN_MAX': {'timeperiod': [5, 10, 20, 40]},
            'TURN_MIN': {'timeperiod': [5, 10, 20, 40]},
            'TURN_STD': {'timeperiod': [5, 10, 20, 40]},
            'TURN_RATE_EMA': {'timeperiod': [5, 10, 20]},
            'TURN_ROC':{'timeperiod': [5, 10, 20, 40]},
            'TURN_SLOPE': {'timeperiod': [5, 10, 20, 40]},
            'TURN_RSI': {'timeperiod': [5, 10, 20, 40]},
            'TURN_TSF': {'timeperiod': [5, 10, 20, 40]},

            'BETA': {},
            
            'AMOUNT_LN': {},
            'AMT_MAX': {'timeperiod': [5, 10, 20,40]},
            'AMT_MIN': {'timeperiod': [5, 10, 20,40]},
            'AMT_STD': {'timeperiod': [5, 10, 20,40]},
            'AMT_EMA': {'timeperiod': [5, 10, 20,40]},
            'AMT_ROC': {'timeperiod': [5, 10, 20, 40]},
            'AMT_TRIX': {'timeperiod': [5, 10, 20, 40]},
            'AMT_SLOPE': {'timeperiod': [5, 10, 20, 40]},
            'AMT_RSI': {'timeperiod': [5, 10, 20, 40]},
            'AMT_TSF': {'timeperiod': [5, 10, 20, 40]},
            'AMT_VAR': {'timeperiod': [5, 10, 20, 40]}
            
            
        }

        self.minimum_data_length = 300 # 最小数据长度,去除交易时间过短的数据，700为三年的时长

        self.median_df = pd.DataFrame()
        self.amount_df = pd.DataFrame()
        self.rank_df = pd.DataFrame()
        self.effective_stock_count_df = pd.DataFrame()
        self.index_code = "sh000300"
        self.index_df = pd.DataFrame(columns=['date', 'pctChg'])

        self.basic_info_df = pd.read_csv(basic_info_path)

        # 去掉code列中的点号，并转换为小写
        self.basic_info_df['code'] = self.basic_info_df['code'].str.replace('.', '').str.lower()
        # 筛选出type为1的股票代码，并转换为集合以提高查找效率
        self.stock_codes_set = set(self.basic_info_df[self.basic_info_df['type'] == 1]['code'])

        self.stock_slice_df = pd.DataFrame()
        self._str_factor_df = pd.DataFrame()

        if stock_pool_path is not None:
            with open(stock_pool_path, 'r') as f:
                self.stock_pool = set()
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        self.stock_pool.add(parts[0].lower())

            print(f"Using stock pool. Loaded {len(self.stock_pool)} stocks from {stock_pool_path}")
        else:
            self.stock_pool = None

    def _create_stock_slice_df(self, stock_data):
        """创建包含日期、代码、涨跌幅和成交量的数据框"""
        print("Creating stock slice DataFrame...")
        dataframes = []
        for file_name, df in stock_data.items():
            if not df.empty and len(df) >= self.minimum_data_length:
                code = file_name.split('.')[0]
                df_reset = df.reset_index()
                df_reset['code'] = code
                dataframes.append(df_reset[['date', 'code', 'pctChg', 'amount']])
        self.stock_slice_df = pd.concat(dataframes).set_index(['date', 'code']).sort_index()

    def pre_process_slice_features(self,stock_data):
        """计算每只股票在每天成交额的占比"""
        self._create_stock_slice_df(stock_data)

        self.amount_df['amount'] = self.stock_slice_df.groupby(level='date')['amount'].sum()
        #print(self.amount_df)

        ## 计算每只股票在当天涨跌幅中的排名，并取对数
        self.rank_df['rank'] = self.stock_slice_df['pctChg'].groupby(level='date').rank(ascending=False)
        #print(self.rank_df.head())
        #                       rank
        #   date        code    
        # 2008-01-02  sh600030    15.0
        #             sh600031    12.0
        #             sh600033     5.0
        #             sh600035     6.0
        #             sh600036    14.0
        #             sh600037     2.0
        #             sh600039     8.0
        #             sh600048     9.0
        #             sh600060     3.0
        #             sh600061    11.0
        #             sh600062    10.0
        #             sh600063     4.0
        #             sh600064     1.0
        #             sh600066    13.0
        #             sh600068     7.0
        
        ##计算总的有效股票数
        self.effective_stock_count_df['count'] = self.stock_slice_df.groupby(level='date')['pctChg'].count()
        #print(self.effective_stock_count_df)
        #              count
        # date             
        # 2008-01-02     15
        # 2008-01-03     16
        # 2008-01-04     17
        # 2008-01-07     17
        # 2008-01-08     17


        ##计算STR凸显性因子
        return_df = pd.DataFrame()
        return_df['pctChg'] = self.stock_slice_df['pctChg']
        # 将 'code' 转换为列，使得每个 'code' 都有一列
        pivot_return_df = return_df.unstack(level='code')
        print("generate STR factors...")
        # 使用新的优化函数来计算每天的STR因子
        self._str_factor_df = self.calculate_daily_str_factors(pivot_return_df)
        #print(_str_factor_df)
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
        print("success")

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

    def _STR_factor(self,
             weight: pd.DataFrame,
             return_df: pd.DataFrame,
             cur_date):
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

    def calculate_daily_str_factors(self, pivot_return_df: pd.DataFrame, n_jobs=-1):
        """
        计算每一天的STR因子。
        
        Args:
            pivot_return_df (pd.DataFrame): 每日回报率数据框，其中索引是日期，列为不同的股票代码。
            
        Returns:
            pd.DataFrame: 每天的STR因子数据框，其中索引是日期，列为不同的股票代码。
        """
        time_range = self.time_range
        # 创建一个空的DataFrame来存储每天的STR因子
        str_factors = pd.DataFrame(index=pivot_return_df.index, columns=pivot_return_df.columns)

        sigma = self._calc_str_sigma(pivot_return_df)
        
        def _calculate_str_factor_for_date(cur_date, sigma,  return_df):
            # 为每一天计算weight

            weight = self._calc_str_weight(sigma=sigma, cur_date=cur_date)
            str_factor = self._STR_factor(weight=weight, return_df=return_df, cur_date=cur_date)
            return str_factor

        # 遍历每一天
        dates_to_process = pivot_return_df.index[time_range:]
        
        # 使用joblib进行并行计算
        results = Parallel(n_jobs=n_jobs)(
            delayed(_calculate_str_factor_for_date)(date, sigma, pivot_return_df)
            for date in tqdm(dates_to_process, desc='Calculating STR factors')
        )
        
        # 将结果存入str_factors DataFrame
        for date, str_factor in zip(dates_to_process, results):
            str_factors.loc[date] = str_factor

        # # 遍历每一天
        # for date in pivot_return_df.index[time_range:]:

        #     # 为每一天计算weight
        #     weight = self._calc_str_weight(sigma=sigma, cur_date=date, time_range=time_range)
        #     str_factor = self._STR_factor(weight=weight, return_df=pivot_return_df, cur_date=date, time_range=time_range)

        #     # 将STR因子存入str_factors DataFrame
        #     str_factors.loc[date] = str_factor
            
        #     #print(str_factors.head(20))

        return str_factors

    def generate_slice_features(self, code):
        """新增基于横截面的特征
        - 成交量占比

        Args:
            code (str): 股票代码，如'sh000001'，sh开头表示上证，sz开头表示深证

        Returns:
            pd.DataFrame: 包含 'DAILY_AMOUNT_RATIO' 列的新 DataFrame，其中索引为日期
        """
        features = {}

        ##step 1: 计算该股票的成交额占当日成交额的比例，并取对数       
        df = self.stock_slice_df.loc[(slice(None), code), ['amount']]
        df = df.reset_index(level='code', drop=True)

        features['DAILY_AMOUNT_RATIO'] = np.log( (df['amount'] + 1) / self.amount_df['amount'] )

        ## step 2:计算STR凸显性因子
        # print(self._str_factor_df.index)
        # print(self._str_factor_df.columns)
        # print(self._str_factor_df.info())
        # # 检查是否存在多级索引
        # if isinstance(self._str_factor_df.index, pd.MultiIndex):
        #     print("\nThe DataFrame has a MultiIndex.")
        #     print("Levels of the MultiIndex:")
        #     for i, level in enumerate(self._str_factor_df.index.levels):
        #         print(f"Level {i}: {level.name} - {level.dtype}")
        
        # if code in self._str_factor_df.columns:
        #     print("Column found and data extracted.")
        # else:
        #     print("Column 'sz002092' not found in the DataFrame.")
            
            
        #str_series = self._str_factor_df[('pctChg', code)]
        #print(str_series)
        
        
        features["STR_FACTOR"] = self._str_factor_df[('pctChg', code)]
        
        ## step 3:计算涨跌幅排名
        rank_series = self.rank_df.loc[(slice(None), code), ['rank']]
        rank_series = rank_series.reset_index(level='code', drop=True)

        features['RANK'] = np.log((rank_series['rank'] + self.effective_stock_count_df['count'] )/ self.effective_stock_count_df['count'])
        #print(features['RANK'])

        return pd.DataFrame(features, index=df.index)

    def generate_single_stock_features(self, df):
        """
        新增更多基于ta-lib的特征
        市值因子
            - size = price*factor*total_share/factor
        价格相关
            - Overlap Studies(重叠指标)
                - EMA,Exponential Moving Average （指数移动平均线）
                        //- BBANDS,Bollinger Bands （布林带）
                - SAR,Parabolic SAR （抛物线转向
                - KAMA, Kaufman Adaptive Moving Average
                - TEMA, Triple Exponential Moving Average
                - TRIMA, Triangular Moving Average
            - Momentum Indicators(动量指标)
                - ADX, Average Directional Movement Index
                - APO, Absolute Price Oscillator
                - AROON, Aroon
                - BOP, Balance Of Power
                - CCI, Commodity Channel Index
                - CMO, Chande Momentum Oscillator
                - MACD, Moving Average Convergence/Divergence
                - MACDEXT, MACD with controllable MA type
                - MOM,Momentum
                - MFI, Money Flow Index
                - ROC, Rate of change : ((price/prevPrice)-1)*100
                - RSI,Relative Strength Index （相对强弱指标）
                - STOCHF, Stochastic Fast
                - STOCHRSI, Stochastic Relative Strength Index
                - TRIX, 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
                - ULTOSC,Ultimate Oscillator
                - WILLR,Williams' %R
            - Volume Indicators(成交量指标)
                - AD, Chaikin A/D Line                   
                - '$TURN_RATE_5',
                - '$TURN_RATE_10',
                - '$TURN_RATE_20',
                - '$TURN_RATE_MIX',
                - '$TURN_ROC_6',
                - '$TURN_ROC_12',
                - '$TURN_ROC_24',
                - '$TURN_ROC_48',
                - '$BETA',
                - ADOSC, Chaikin A/D Oscillator
                - OBV, On Balance Volume
            - Volatility Indicators(波动率指标)
                - ATR, Average True Range
                - NATR, Normalized Average True Range
                - TRANGE, True Range
            - Statistic Functions(统计函数)
                - LINEARREG_SLOPE - v4.1, Linear Regression Slope 
                - TSF - v4.1, Time Series Forecast
                - VAR - v4.1, Variance
            - beta 指标
        - Turnover Rate 换手率相关指标
            - Log, TURN_RATE_LN,v4.1, 成交量的对数
            Math, 数学运算
                - TURN_MAX, v4.1, 成交量的对数的最大值
                - TURN_MIN, 成交量的对数的最小值
                - TURN_STD, 成交量的对数的标准差
            Overlap/Momentum/Statistics
            - EMA, TURN_RATE_EMA
            - ROC, TURN_ROC_
            - SLOPE, TURN_SLOPE_ , v4.1
            - RSI, TURN_RSI_, v4.1
            - TSF, TURN_TSF_, v4.1
        - Amount，成交量相关指标,v4.1, 
            - AMOUNT_LN, 成交量的对数
            Math, 数学运算
                - AMT_MAX, 成交量的对数的最大值
                - AMT_MIN, 成交量的对数的最小值
                - AMT_STD, 成交量的对数的标准差
            Overlap/Momentum/Statistics
            - AMT_EMA
            - AMT_ROC
            - AMT_TRIX
            - AMT_SLOPE, Linear Regression Slope
            - AMT_RSI
            - AMT_TSF
            - AMT_VAR
        Args:
            df (_type_): _description_

        Returns:
            _type_: _description_
        """

        features = {}

        # 确保数据列名符合预期
        open_col = 'open' if 'open' in df.columns else 'Open'
        close_col = 'close' if 'close' in df.columns else 'Close'
        high_col = 'high' if 'high' in df.columns else 'High'
        low_col = 'low' if 'low' in df.columns else 'Low'
        volume_col = 'volume' if 'volume' in df.columns else 'Volume'
        turn_col = 'turn' if 'turn' in df.columns else 'Turnover'
        pct_chg_col = 'pctChg' if 'pctChg' in df.columns else 'pct_chg'
        amount_col = 'amount' if 'amount' in df.columns else 'Amount'
        factor_col = 'factor' if 'factor' in df.columns else 'Factor'
        totalShare_col = 'totalShare' if 'totalShare' in df.columns else 'TotalShare'
        
        df[factor_col] = df[factor_col].ffill().fillna(1)
        
        ## 市值因子
        features['SIZE'] = np.log(df[open_col] * (df[totalShare_col]+1) / df[factor_col])
        

        # EMA,Exponential Moving Average （指数移动平均线）
        for period in self.feature_functions['EMA']['timeperiod']:
            features[f'EMA_{period}'] = talib.EMA(df[open_col], timeperiod=period)

        # SAR,Parabolic SAR （抛物线转向）
        features['SAR'] = talib.SAR(df[high_col], df[low_col])

        # KAMA, Kaufman Adaptive Moving Average
        for period in self.feature_functions['KAMA']['timeperiod']:
            features[f'KAMA_{period}'] = talib.KAMA(df[open_col], timeperiod=period)

        # TEMA, 
        for period in self.feature_functions['TEMA']['timeperiod']:
            features[f'TEMA_{period}'] = talib.TEMA(df[open_col], timeperiod=period)

        # TRIMA
        for period in self.feature_functions['TRIMA']['timeperiod']:
            features[f'TRIMA_{period}'] = talib.TRIMA(df[open_col], timeperiod=period)
            
        
        # ADX, Average Directional Movement Index
        for period in self.feature_functions['ADX']['timeperiod']:
            features[f'ADX_{period}'] = talib.ADX(df[high_col], df[low_col], df[open_col], timeperiod=period)

        # APO
        features['APO'] = talib.APO(df[close_col])
        
        # AROON
        for period in self.feature_functions['AROON']['timeperiod']:
            features[f'AROON_{period}_down'], features[f'AROON_{period}_up'] = talib.AROON(df[high_col], df[low_col], timeperiod=period)

        # BOP, Balance Of Power
        features['BOP'] = talib.BOP(df[open_col], df[high_col], df[low_col], df[close_col])

        # CCI, Commodity Channel Index
        for period in self.feature_functions['CCI']['timeperiod']:
            features[f'CCI_{period}'] = talib.CCI(df[high_col], df[low_col], df[open_col], timeperiod=period)
        
        # CMO
        for period in self.feature_functions['CMO']['timeperiod']:
            features[f'CMO_{period}'] = talib.CMO(df[open_col], timeperiod=period)
        
        # MACD, Moving Average Convergence/Divergence
        macd, signal, hist = talib.MACD(df[open_col],
                                      fastperiod=self.feature_functions['MACD']['fastperiod'],
                                      slowperiod=self.feature_functions['MACD']['slowperiod'],
                                      signalperiod=self.feature_functions['MACD']['signalperiod'])
        features['MACD'] = macd
        features['MACD_SIGNAL'] = signal
        features['MACD_HIST'] = hist

        # MACDEXT, MACD with controllable MA type
        #macd, signal, hist = talib.MACDEXT(df[close_col])

        # MOM,Momentum
        for period in self.feature_functions['MOM']['timeperiod']:
            features[f'MOM_{period}'] = talib.MOM(df[open_col], timeperiod=period)

        # MFI
        for period in self.feature_functions['MFI']['timeperiod']:
            features[f'MFI_{period}'] = talib.MFI(df[high_col], df[low_col], df[open_col], df[volume_col], timeperiod=period)

        #ROC
        for period in self.feature_functions['ROC']['timeperiod']:
            features[f'ROC_{period}'] = talib.ROC(df[open_col], timeperiod=period)

        # RSI,Relative Strength Index （相对强弱指标）
        for period in self.feature_functions['RSI']['timeperiod']:
            features[f'RSI_{period}'] = talib.RSI(df[open_col], timeperiod=period)
            
        # STOCHF
        fastk, fastd = talib.STOCHF(df[high_col], df[low_col], df[open_col])
        features['STOCHF_k'] = fastk
        features['STOCHF_d'] = fastd
        
        #STOCHRSI
        stochrsi_fastk, stochrsi_fastd = talib.STOCHRSI(df[open_col])
        features['STOCHRSI_k'] = stochrsi_fastk
        features['STOCHRSI_d'] = stochrsi_fastd
        
        #TRIX
        for period in self.feature_functions['TRIX']['timeperiod']:
            features[f'TRIX_{period}'] = talib.TRIX(df[open_col], timeperiod=period)


        # ULTOSC,Ultimate Oscillator
        features['ULTOSC'] = talib.ULTOSC(df[high_col], df[low_col], df[open_col])

        # WILLR,Williams' %R
        for period in self.feature_functions['WILLR']['timeperiod']:
            features[f'WILLR_{period}'] = talib.WILLR(df[high_col], df[low_col], df[open_col], timeperiod=period)

        # AD, Chaikin A/D Line
        features['AD'] = talib.AD(df[high_col], df[low_col], df[open_col], df[volume_col])

        # ADOSC, Chaikin A/D Oscillator
        features['ADOSC'] = talib.ADOSC(df[high_col], df[low_col], df[open_col], df[volume_col], 
                                        fastperiod=self.feature_functions['ADOSC']['fastperiod'], 
                                        slowperiod=self.feature_functions['ADOSC']['slowperiod'])
        # 生成OBV
        features['OBV'] = talib.OBV(df[open_col], df[volume_col])

        # ATR, Average True Range
        for period in self.feature_functions['ATR']['timeperiod']:
            features[f'ATR_{period}'] = talib.ATR(df[high_col], df[low_col], df[open_col], timeperiod=period)

        # NATR, Normalized Average True Range
        for period in self.feature_functions['NATR']['timeperiod']:
            features[f'NATR_{period}'] = talib.NATR(df[high_col], df[low_col], df[open_col], timeperiod=period)

        # TRANGE, True Range
        features['TRANGE'] = talib.TRANGE(df[high_col], df[low_col], df[open_col])

        # LINEARREG_SLOPE - v4.1, Linear Regression Slope
        for period in self.feature_functions['LINEARREG_SLOPE']['timeperiod']:
            features[f'LINEARREG_SLOPE_{period}'] = talib.LINEARREG_SLOPE(df[open_col], timeperiod=period)
        
        for period in self.feature_functions['TSF']['timeperiod']:
            features[f'TSF_{period}'] = talib.SUB(df[open_col], talib.TSF(df[open_col], timeperiod=period))
        
        for period in self.feature_functions['VAR']['timeperiod']:
            features[f'VAR_{period}'] = talib.VAR(df[open_col], timeperiod=period)
        
        # ## beta 指标,
        # comm_dates = df.index
        # aligned_index_df = self.index_df.loc[comm_dates]
        # aligned_index_df.fillna(0, inplace=True)
        # features['BETA'] = talib.BETA(df[pct_chg_col], aligned_index_df[pct_chg_col], timeperiod=40)
    
        
        ## Amount，成交量相关指标
        df['TURN_RATE_LN'] = np.log(df[turn_col]+0.00001)
        turn_ln_col = 'TURN_RATE_LN'
        ## AMOUNT_LN, 成交量的对数
        features[turn_ln_col] = df[turn_ln_col]
        
        ## Math, 数学运算
        ## TURN_MAX, 成交量的对数的最大值
        for period in self.feature_functions['TURN_MAX']['timeperiod']:
            features[f'TURN_MAX_{period}'] = talib.MAX(df[turn_ln_col], timeperiod=period)
        
        ## TURN_MIN, 成交量的对数的最小值
        for period in self.feature_functions['TURN_MIN']['timeperiod']:
            features[f'TURN_MIN_{period}'] = talib.MIN(df[turn_ln_col], timeperiod=period)
        
        ## TURN_STD, 成交量的对数的标准差
        #for period in self.feature_functions['TURN_STD']['timeperiod']:
        #    features[f'TURN_STD_{period}'] = talib.STDDEV(df[turn_ln_col], timeperiod=period)
            
        # EMA
        for period in self.feature_functions['TURN_RATE_EMA']['timeperiod']:
            features[f'TURN_RATE_EMA_{period}'] = talib.EMA(df[turn_ln_col], timeperiod=period)

        ## Turnover Rate ROC
        for period in self.feature_functions['TURN_ROC']['timeperiod']:
            features[f'TURN_ROC_{period}'] = talib.ROC(df[turn_ln_col], timeperiod=period)

        ## Turnover Rate SLOPE
        for period in self.feature_functions['TURN_SLOPE']['timeperiod']:
            features[f'TURN_SLOPE_{period}'] = talib.LINEARREG_SLOPE(df[turn_ln_col], timeperiod=period)
        
        ## Turnover Rate RSI
        for period in self.feature_functions['TURN_RSI']['timeperiod']:
            features[f'TURN_RSI_{period}'] = talib.RSI(df[turn_ln_col], timeperiod=period)
          
        ## Turnover Rate TSF
        for period in self.feature_functions['TURN_TSF']['timeperiod']:
            features[f'TURN_TSF_{period}'] = talib.SUB(df[turn_ln_col], talib.TSF(df[turn_ln_col], timeperiod=period))
            

            
        ## Amount，成交量相关指标
        df['AMOUNT_LN'] = np.log(df[amount_col]+1)
        amount_ln_col = 'AMOUNT_LN'
        ## AMOUNT_LN, 成交量的对数
        features[amount_ln_col] = df[amount_ln_col]
        
        ## Math, 数学运算
        ## AMT_MAX, 成交量的对数的最大值
        for period in self.feature_functions['AMT_MAX']['timeperiod']:
            features[f'AMT_MAX_{period}'] = talib.MAX(df[amount_ln_col], timeperiod=period)
        
        ## AMT_MIN, 成交量的对数的最小值
        for period in self.feature_functions['AMT_MIN']['timeperiod']:
            features[f'AMT_MIN_{period}'] = talib.MIN(df[amount_ln_col], timeperiod=period)
        
        ## AMT_STD, 成交量的对数的标准差
        #for period in self.feature_functions['AMT_STD']['timeperiod']:
        #    features[f'AMT_STD_{period}'] = talib.STDDEV(df[amount_ln_col], timeperiod=period)
        
        ## Overlap/Momentum/Statistics
        ## AMT_EMA
        for period in self.feature_functions['AMT_EMA']['timeperiod']:
            features[f'AMT_EMA_{period}'] = talib.EMA(df[amount_ln_col], timeperiod=period)
        
        ## AMT_ROC
        for period in self.feature_functions['AMT_ROC']['timeperiod']:
            features[f'AMT_ROC_{period}'] = talib.ROC(df[amount_ln_col], timeperiod=period)

        ## AMT_TRIX
        for period in self.feature_functions['AMT_TRIX']['timeperiod']:
            features[f'AMT_TRIX_{period}'] = talib.TRIX(df[amount_ln_col], timeperiod=period)
        
        ## AMT_SLOPE, Linear Regression Slope
        for period in self.feature_functions['AMT_SLOPE']['timeperiod']:
            features[f'AMT_SLOPE_{period}'] = talib.LINEARREG_SLOPE(df[amount_ln_col], timeperiod=period)
        
        ## AMT_RSI
        for period in self.feature_functions['AMT_RSI']['timeperiod']:
            features[f'AMT_RSI_{period}'] = talib.RSI(df[amount_ln_col], timeperiod=period)
        
        ## AMT_TSF
        for period in self.feature_functions['AMT_TSF']['timeperiod']:
            features[f'AMT_TSF_{period}'] = talib.SUB( df[amount_ln_col] ,talib.TSF(df[amount_ln_col], timeperiod=period))
        
        ## AMT_VAR
        for period in self.feature_functions['AMT_VAR']['timeperiod']:
            features[f'AMT_VAR_{period}'] = talib.VAR(df[amount_ln_col], timeperiod=period)
        return pd.DataFrame(features, index=df.index)
    
    def generate_rsrs_features(self, df):
        """
        新增RSRS特征
        Args:
            df (_type_): _description_

        Returns:
            _type_: _description_
        """
        features = {}
        # 确保数据列名符合预期
        assert 'high' in df.columns
        assert 'low' in df.columns
        df = df[['high', 'low']].copy()

        ## RSRS
        return RSRSFeature.calculate_rsrs_features(df=df, time_range=20,window_size=500) ##2年的数据作为时间窗口
        
        

    def _process_stock_data(self, file_name, df, output_dir):
        code = file_name.split('.')[0]
        # 生成新特征
        new_features = self.generate_single_stock_features(df)
        slice_features = self.generate_slice_features(code)
        rsrs_features = self.generate_rsrs_features(df)
        # 合并特征
        #result = pd.concat([df, new_features, slice_features], axis=1)
        result = pd.concat([df, new_features, slice_features,rsrs_features], axis=1)
        output_path = Path(output_dir) / file_name
        result.to_csv(output_path)
        #columns = new_features.columns.tolist() + slice_features.columns.tolist()
        columns = new_features.columns.tolist() + slice_features.columns.tolist() + rsrs_features.columns.tolist()
        return columns

    def process_directory(self, input_dir, output_dir,feature_meta_file = None):
        """处理整个目录的CSV文件"""
        
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 读取所有CSV文件
        print("Reading files for 1st round...")
        stock_data = {}
        non_stock_data = {}
        effective_count = 0
        empty_count = 0
        short_count = 0
        csv_files = Path(input_dir).glob('*.csv')
        print(f"{input_dir} have {len(list(csv_files))} files")
        for file_path in Path(input_dir).glob('*.csv'):
            filename = file_path.name
            stock_id = filename.rsplit('.', 1)[0]
        

            if any(code in filename for code in self.stock_codes_set): # 如果是股票
                try:
                    if self.stock_pool is not None and stock_id not in self.stock_pool:
                        continue
                     # 读取CSV文件,索引列为第一列，
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
### 读取的数据格式如下：
# date,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,isST,peTTM,pbMRQ,psTTM,pcfNcfTTM,foreAdjustFactor,backAdjustFactor,adjustFactor,factor,vwap,epsTTM,totalShare,liqaShare
# 2008-01-02,44.80247325,45.67666785,43.78257955,45.4581192,44.36537595,9930800.0,61302049.0,2.203873,1.0,2.463048,0.0,1791.807669,4.420813,1.68484,-18.929168,1.0,1.0,1.0,1.0,6.172921516896927,0.003483,623700000.00,450606738.00
# 2008-01-03,45.38526965,46.9151102,44.94817235,46.4780129,45.4581192,15608075.0,98710863.0,3.463791,1.0,2.243595,0.0,1832.008482,4.519998,1.722641,-19.353861,1.0,1.0,1.0,1.0,6.3243457633308395,0.003483,623700000.00,450606738.00
                    if df.empty: # 去除空数据
                        print(f"Empty file: {file_path.name}")
                        empty_count += 1
                        continue

                    # 如果数据长度不足，则跳过
                    if len(df) < self.minimum_data_length:
                        #print(f"Data too short: {file_path.name}")
                        short_count += 1
                        continue

                    stock_data[file_path.name] = df
                    effective_count += 1
                except Exception as e:
                    print(f"Error reading {file_path.name}: {str(e)}")
            else:
                #print(f"Reading non-stock file: {file_path.name}")
                df = pd.read_csv(file_path, parse_dates=['date'])
                non_stock_data[file_path.name] = df

            if self.index_code in filename: # 如果是指数
                print(f"Reading index file: {file_path.name}")
                df = pd.read_csv(file_path, parse_dates=['date'])

                # 提取需要的列
                self.index_df = df[['date', 'pctChg']]
                self.index_df.set_index('date', inplace=True)
        total_count = len(list(Path(input_dir).glob('*.csv')))
        print(f"Total files: {total_count}, Stock files: {len(stock_data)}, Non-stock files: {len(non_stock_data)}")
        print(f"Effective files: {effective_count}, Empty files: {empty_count}, Short files: {short_count}")

        ##计算横截面特征，需要先计算全局数据
        self.pre_process_slice_features(stock_data)
        feature_names = set()
        # 处理每个CSV文件
        # 
        # for file_name, df in tqdm(stock_data.items(), desc="Process stock data...", unit="file"):
        #     try:
        #         cols_new = self._process_stock_data(file_name, df, output_dir)
        #         feature_names.update(cols_new)
        #     except Exception as e:
        #         print(f"Error processing {file_name}: {str(e)}")
                
        # 使用joblib进行并行计算
        results = Parallel(n_jobs=-1)(
            delayed(self._process_stock_data)(file_name, df, output_dir)
            for file_name, df in tqdm(stock_data.items(), desc="Process stock data...", unit="file")
        )

        # 更新特征名称
        for cols_new in results:
            feature_names.update(cols_new)
        
        # with ProcessPoolExecutor() as executor:
        #     futures = {}
        #     for file_name, df in stock_data.items():
        #         futures[executor.submit(self._process_stock_data, file_name, df, output_dir)] = file_name

        #     for future in tqdm(as_completed(futures), desc="Process stock data...", unit="file"):
        #         file_name = futures[future]
        #         try:
        #             cols_new = future.result()
        #             feature_names.update(cols_new)
        #         except Exception as e:
        #             print(f"Error processing {file_name}: {str(e)}")

        # 生成feature dict
        feature_meta_dic = {}
        feature_meta_dic['fields'] = []
        feature_meta_dic['names'] = []
        feature_meta_dic['description'] = "version: v4.1, code time: 2025-02-01, scope: ta-lib/STR/RSRS, feature count: %d" % len(feature_names)
        for feature_name in sorted(feature_names):
            feature_meta_dic['fields'].append(f"${feature_name}")
            feature_meta_dic['names'].append(feature_name)

        # 输出特征名称到文件
        if feature_meta_file is None:
            meta_path = Path(output_dir) / 'feature_names.json'
        else:
            meta_path = feature_meta_file
            
        ##将dict 输出到json文件中
        with open(meta_path, 'w') as f:
            json.dump(feature_meta_dic, f, indent=4)

        for file_name, df in tqdm(non_stock_data.items(), desc="Process non-stock data...", unit="file"):
            # 仅保存沪深300的数据
            if self.index_code in file_name:
                output_path = Path(output_dir) / file_name
                df.to_csv(output_path)

    def _check_file_status(self, input_path: Path, output_path: Path) -> dict:
        """检查文件状态，判断是否需要更新
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            
        Returns:
            dict: {
                'needs_update': bool,  # 是否需要更新
                'last_date': pd.Timestamp,  # 最后更新日期
                'history_start': pd.Timestamp,  # 需要的历史数据起始日期
                'input_df': pd.DataFrame,  # 输入数据
                'existing_df': pd.DataFrame,  # 已存在的数据
            }
        """
        max_history_window = max(48, self.time_range)  # EMA=48, STR=time_range
        
        # 读取输入文件
        input_df = pd.read_csv(input_path, index_col=0, parse_dates=True)
        
        if not output_path.exists():
            return {
                'needs_update': True,
                'last_date': None,
                'history_start': None,
                'input_df': input_df,
                'existing_df': None
            }
            
        # 读取已存在的输出文件
        existing_df = pd.read_csv(output_path, index_col=0, parse_dates=True)
        last_date = existing_df.index.max()
        
        # 检查是否需要更新
        needs_update = input_df.index.max() > last_date
        
        if needs_update:
            history_start = last_date - pd.Timedelta(days=max_history_window)
        else:
            history_start = None
            
        return {
            'needs_update': needs_update,
            'last_date': last_date,
            'history_start': history_start,
            'input_df': input_df,
            'existing_df': existing_df
        }

    def _process_stock_data_incremental(self, file_name: str, df: pd.DataFrame, 
                                      output_path: Path, last_date=None):
        """增量处理单个股票数据"""
        try:
            code = file_name.split('.')[0]
            
            # 生成新特征
            new_features = self.generate_single_stock_features(df)
            slice_features = self.generate_slice_features(code)
            rsrs_features = self.generate_rsrs_features(df)
            
            # 合并特征
            result = pd.concat([df, new_features, slice_features, rsrs_features], axis=1)
            
            # 如果是增量更新，合并新旧数据
            if last_date is not None:
                # 读取已有数据
                existing_df = pd.read_csv(output_path, index_col=0, parse_dates=True)
                # 保留旧数据中不需要更新的部分
                old_data = existing_df[existing_df.index <= last_date]
                # 从新计算的结果中只保留需要更新的部分
                new_data = result[result.index > last_date]
                # 合并新旧数据
                result = pd.concat([old_data, new_data])
                
            # 保存结果
            result.to_csv(output_path)
            return result.columns.tolist()
            
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            return []

    def process_directory_incremental(self, input_dir, output_dir, feature_meta_file=None):
        """增量处理整个目录的CSV文件"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print("Scanning files...")
        stock_data = {}
        update_files = []
        effective_count = 0
        empty_count = 0
        short_count = 0

        # 1. 扫描文件并检查更新状态
        for file_path in Path(input_dir).glob('*.csv'):
            filename = file_path.name
            output_path = Path(output_dir) / filename

            if any(code in filename for code in self.stock_codes_set):
                try:
                    status = self._check_file_status(file_path, output_path)

                    if status['input_df'].empty:
                        empty_count += 1
                        continue

                    if len(status['input_df']) < self.minimum_data_length:
                        short_count += 1
                        continue

                    if status['needs_update']:
                        stock_data[filename] = status['input_df']
                        update_files.append({
                            'filename': filename,
                            'status': status,
                            'output_path': output_path
                        })
                        effective_count += 1

                except Exception as e:
                    print(f"Error checking {filename}: {e}")
                    
        print(f"Found {len(update_files)} files to update")
        
        if not update_files:
            print("No files need to be updated.")
            return
            
        # 2. 计算横截面特征
        print("Calculating cross-sectional features...")
        self.pre_process_slice_features(stock_data)
        
        # 3. 并行处理需要更新的文件
        print("Processing updates...")
        results = Parallel(n_jobs=-1)(
            delayed(self._process_stock_data_incremental)(
                file_info['filename'],
                stock_data[file_info['filename']],
                file_info['output_path'],
                file_info['status']['last_date']
            )
            for file_info in tqdm(update_files, desc="Updating files")
        )
        
        print(f"Successfully processed {len(results)} files")

def __test__():
    basic_info_path = '/home/godlike/project/GoldSparrow/Day_Data/qlib_data/basic_info.csv'
    in_folder = '/home/godlike/project/GoldSparrow/Day_Data/test_raw'
    out_folder = '/home/godlike/project/GoldSparrow/Day_Data/test_raw_ta'
    feature_meta_file = '/home/godlike/project/GoldSparrow/Day_Data/feature_names.json'
    stock_pool_file = '/home/godlike/project/GoldSparrow/Day_Data/qlib_data/instruments/csi300.txt'

    feature_generator = TALibFeatureExt(
        basic_info_path=basic_info_path,
        time_range=30,
        stock_pool_path=stock_pool_file
    )
    # 使用增量更新方法
    feature_generator.process_directory_incremental(in_folder, out_folder, feature_meta_file)

if __name__ == '__main__':
    __test__()