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
from cpv_feature import CPVFeature
from str_feature import STRFeature



class TALibFeatureExt:
    """_summary_
    """
    def __init__(self, basic_info_path, time_range=30, stock_pool_path=None, cpv_feature_config_path=None, rsrs_cache_path=None, n_jobs=-1):
        """初始化特征生成器"""
        # 定义要生成的技术指标及其参数
        self.window_size_global = max(time_range, 60) ###time_range是STR因子所需的时间窗口，48是EMA等ta-lib因子所需的时间窗口,取60是一个更加安全的边界
        self.window_size_rsrs = 550 ## RSRS因子需要500天的时间窗口+50天的time_range
        
        self.time_range = time_range
        
        self.cpv_feature_config_path = cpv_feature_config_path
        if self.cpv_feature_config_path is not None:
            print( f"Using CPV feature, configure file: {self.cpv_feature_config_path}")
            self.cpv_feature_cls = CPVFeature(self.cpv_feature_config_path)
            self.cpv_df = self.cpv_feature_cls.online_process()
        else:
            self.cpv_feature_cls = None
            self.cpv_df = None
        
        self.n_jobs = n_jobs
        
        self.rsrs_feature_cls = RSRSFeature(time_range=20, window_size=500, cache_path=rsrs_cache_path)
        self.str_feature_cls = STRFeature(time_range=time_range, n_jobs=n_jobs)

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
        self.index_df = pd.DataFrame()

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
        """创建包含日期、代码、涨跌幅和成交量的数据框
        数据样例：
                              pctChg    amount
        date        code                       
        2023-01-01  000001     1.2      100000
                    000002     0.8      120000
        2023-01-02  000001    -0.5       80000
                    000002     1.5       90000
        """
        print("Creating stock slice DataFrame...")
        dataframes = []
        for code, df in stock_data.items():
            if not df.empty:
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
        #print(pivot_return_df)
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
# 2025-02-10   3.5821  -0.6630   1.4228   2.0761   0.6623      NaN   3.2680  -0.2801   3.6187   0.7299      NaN  ...  -0.5642   0.5038      NaN      NaN   6.4815  -0.1288  -0.9756   2.1423      NaN   0.5521      NaN

        print("generate STR factors...")
        # 使用新的优化函数来计算每天的STR因子
        self._str_factor_df = self.str_feature_cls.calculate_str_features(pivot_return_df)
        #print(self._str_factor_df)  # Include this line to check the generated STR factors
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

        str_series = self._str_factor_df[('pctChg', code)]

        # ## object对象，导致后续错误，debug其中的原因
        # print(str_series.dtypes)
        # #(a) 检查列中的唯一值
        # print(str_series.unique())
        # #  [nan 0.1392545056732077 0.1579105844503957 0.07848061706044696
        # #  0.06342289740362128 0.03961372798913102 0.07275462601001799
        # #  0.061139080350342906 0.06579270155596961 -0.0067452817056449545
        # #  0.11665692584030171 0.05213464940148926 -0.02388937946936449
        # #  -0.056990625291462454 0.7655954345992414 0.7534942850249533
        # #  0.7883460276467572 0.7742736673486761 0.7479054342338002
        # #  0.8003399995687005 0.8959945560724697 0.8775964181158891
        # #  0.9420927766658975 0.9153309509933375 0.9490285167653411
        # #  0.8888593326544392 0.8578108862074821 0.9112028759061824
        # #  0.8235432169897992 0.9252993605752655 0.8681914564561273
        # #  0.8819553958457887 0.9061086749153077 0.9789989556008282
        # #  0.8807629972296485 0.8903476271546746]

        # #(b) 检查是否有非数值数据
        # print(str_series.apply(lambda x: isinstance(x, str)).any())  # 是否有字符串—————— False
        # print(str_series.apply(lambda x: isinstance(x, (int, float))).all())  # 是否全是数值 ———— True

        # # (c) 检查是否有 NaN 或 None
        # print(str_series.isnull().sum())  # 缺失值数量———— 30

        #print(str_series)

        str_series = pd.to_numeric(str_series, errors='coerce')
        #print(str_series.dtype)  # 应输出 float64

        features["STR_FACTOR"] = str_series # self._str_factor_df[('pctChg', code)]

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
        
        ## Amount，成交量相关指标
        
        turn_rate_df = pd.DataFrame()
        turn_rate_df['TURN_RATE_LN'] = np.log(df[turn_col] + 0.00001)
        turn_ln_col = 'TURN_RATE_LN'
        ## AMOUNT_LN, 成交量的对数
        features[turn_ln_col] = turn_rate_df['TURN_RATE_LN']
        
        ## Math, 数学运算
        ## TURN_MAX, 成交量的对数的最大值
        for period in self.feature_functions['TURN_MAX']['timeperiod']:
            features[f'TURN_MAX_{period}'] = talib.MAX(turn_rate_df[turn_ln_col], timeperiod=period)
        
        ## TURN_MIN, 成交量的对数的最小值
        for period in self.feature_functions['TURN_MIN']['timeperiod']:
            features[f'TURN_MIN_{period}'] = talib.MIN(turn_rate_df[turn_ln_col], timeperiod=period)
        
        ## TURN_STD, 成交量的对数的标准差
        #for period in self.feature_functions['TURN_STD']['timeperiod']:
        #    features[f'TURN_STD_{period}'] = talib.STDDEV(df[turn_ln_col], timeperiod=period)
            
        # EMA
        for period in self.feature_functions['TURN_RATE_EMA']['timeperiod']:
            features[f'TURN_RATE_EMA_{period}'] = talib.EMA(turn_rate_df[turn_ln_col], timeperiod=period)

        ## Turnover Rate ROC
        for period in self.feature_functions['TURN_ROC']['timeperiod']:
            features[f'TURN_ROC_{period}'] = talib.ROC(turn_rate_df[turn_ln_col], timeperiod=period)

        ## Turnover Rate SLOPE
        for period in self.feature_functions['TURN_SLOPE']['timeperiod']:
            features[f'TURN_SLOPE_{period}'] = talib.LINEARREG_SLOPE(turn_rate_df[turn_ln_col], timeperiod=period)
        
        ## Turnover Rate RSI
        for period in self.feature_functions['TURN_RSI']['timeperiod']:
            features[f'TURN_RSI_{period}'] = talib.RSI(turn_rate_df[turn_ln_col], timeperiod=period)
          
        ## Turnover Rate TSF
        for period in self.feature_functions['TURN_TSF']['timeperiod']:
            features[f'TURN_TSF_{period}'] = talib.SUB(turn_rate_df[turn_ln_col], talib.TSF(turn_rate_df[turn_ln_col], timeperiod=period))

        ## AMOUNT_LN, 成交量的对数
        amount_rate_df = pd.DataFrame()
        amount_ln_col = 'AMOUNT_LN'
        amount_rate_df[amount_ln_col] = np.log(df[amount_col]+1)
        features[amount_ln_col] = amount_rate_df[amount_ln_col]
        
        ## Math, 数学运算
        ## AMT_MAX, 成交量的对数的最大值
        for period in self.feature_functions['AMT_MAX']['timeperiod']:
            features[f'AMT_MAX_{period}'] = talib.MAX(amount_rate_df[amount_ln_col], timeperiod=period)
        
        ## AMT_MIN, 成交量的对数的最小值
        for period in self.feature_functions['AMT_MIN']['timeperiod']:
            features[f'AMT_MIN_{period}'] = talib.MIN(amount_rate_df[amount_ln_col], timeperiod=period)
        
        ## AMT_STD, 成交量的对数的标准差
        #for period in self.feature_functions['AMT_STD']['timeperiod']:
        #    features[f'AMT_STD_{period}'] = talib.STDDEV(amount_rate_df[amount_ln_col], timeperiod=period)
        
        ## Overlap/Momentum/Statistics
        ## AMT_EMA
        for period in self.feature_functions['AMT_EMA']['timeperiod']:
            features[f'AMT_EMA_{period}'] = talib.EMA(amount_rate_df[amount_ln_col], timeperiod=period)
        
        ## AMT_ROC
        for period in self.feature_functions['AMT_ROC']['timeperiod']:
            features[f'AMT_ROC_{period}'] = talib.ROC(amount_rate_df[amount_ln_col], timeperiod=period)

        ## AMT_TRIX
        for period in self.feature_functions['AMT_TRIX']['timeperiod']:
            features[f'AMT_TRIX_{period}'] = talib.TRIX(amount_rate_df[amount_ln_col], timeperiod=period)
        
        ## AMT_SLOPE, Linear Regression Slope
        for period in self.feature_functions['AMT_SLOPE']['timeperiod']:
            features[f'AMT_SLOPE_{period}'] = talib.LINEARREG_SLOPE(amount_rate_df[amount_ln_col], timeperiod=period)
        
        ## AMT_RSI
        for period in self.feature_functions['AMT_RSI']['timeperiod']:
            features[f'AMT_RSI_{period}'] = talib.RSI(amount_rate_df[amount_ln_col], timeperiod=period)
        
        ## AMT_TSF
        for period in self.feature_functions['AMT_TSF']['timeperiod']:
            features[f'AMT_TSF_{period}'] = talib.SUB( amount_rate_df[amount_ln_col] ,talib.TSF(amount_rate_df[amount_ln_col], timeperiod=period))
        
        ## AMT_VAR
        for period in self.feature_functions['AMT_VAR']['timeperiod']:
            features[f'AMT_VAR_{period}'] = talib.VAR(amount_rate_df[amount_ln_col], timeperiod=period)

        return pd.DataFrame(features, index=df.index)
    
    def generate_rsrs_features(self, df, stock_id):
        """
        新增RSRS特征
        Args:
            df (_type_): _description_

        Returns:
            _type_: _description_
        """
        # 确保数据列名符合预期
        assert 'high' in df.columns
        assert 'low' in df.columns

        ## RSRS
        return self.rsrs_feature_cls.calculate_rsrs_features(df=df, stock_id=stock_id)

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
                'history_rsrs_start': pd.Timestamp,  # RSRS因子所需的历史数据起始日期
                'input_df': pd.DataFrame,  # 输入数据
                'existing_df': pd.DataFrame,  # 已存在的数据
            }
        """
        # 读取输入文件
        input_df = pd.read_csv(input_path, index_col=0, parse_dates=True)
        df_start_date = input_df.index.min()
        df_length = len(input_df)

        if not output_path.exists():
            rsrs_input_df = input_df.copy()
            return {
                'needs_update': True,
                'last_date': None,
                'df_length':df_length,
                'input_df': input_df,
                'input_rsrs_df': rsrs_input_df
            }

        # 读取已存在的输出文件
        existing_df = pd.read_csv(output_path, index_col=0, parse_dates=True)
        last_date = existing_df.index.max()

        # 检查是否需要更新
        needs_update = input_df.index.max() > last_date

        if needs_update:
            pos = input_df.index.get_loc(last_date)
            start_idx = max(0, pos - self.window_size_global)
            history_start = input_df.index[start_idx]
            df_global_cast = input_df[input_df.index >= history_start].copy() ## 截断dataframe
            
            pos_rsrs = input_df.index.get_loc(last_date)
            start_idx_rsrs = max(0, pos_rsrs - self.window_size_rsrs)
            history_rsrs_start = input_df.index[start_idx_rsrs]
            df_rsrs_cast = input_df.loc[history_rsrs_start:, ['high', 'low']].copy()  ## 截断dataframe
        else:
            history_start = None
            history_rsrs_start = None
            df_global_cast = None
            df_rsrs_cast = None

        return {
            'needs_update': needs_update,
            'last_date': last_date,
            'df_length':df_length,
            'input_df': df_global_cast,
            'input_rsrs_df': df_rsrs_cast            
        }

    def _process_stock_data_incremental(self, stock_id: str, df: pd.DataFrame,
                                        rsrs_df: pd.DataFrame,
                                        cpv_df: pd.DataFrame,
                                      output_path: Path, last_date=None):
        """增量处理单个股票数据"""
        try:
            code = stock_id
            
            if df.empty or rsrs_df.empty:
                raise ValueError(f"Empty DataFrame for {code}")

            if last_date is not None:
                # 生成新特征
                new_features = self.generate_single_stock_features(df)
                #new_features = new_features[new_features.index > last_date]  一次性对齐，不需要截断
                #print(new_features.columns.to_list())
                #['SIZE', 'EMA_5', 'EMA_10', 'EMA_20', 'SAR', 'KAMA_12', 'KAMA_24', 'KAMA_48', 'TEMA_12', 'TEMA_24', 'TEMA_48', 'TRIMA_12', 'TRIMA_24', 'TRIMA_48', 'ADX_14', 'ADX_28', 'APO', 'AROON_14_down', 'AROON_14_up', 'AROON_28_down', 'AROON_28_up', 'BOP', 'CCI_14', 'CCI_28', 'CMO_14', 'CMO_28', 'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'MOM_6', 'MOM_12', 'MOM_24', 'MOM_48', 'MFI_6', 'MFI_12', 'MFI_24', 'MFI_48', 'ROC_6', 'ROC_12', 'ROC_24', 'ROC_48', 'RSI_6', 'RSI_12', 'RSI_24', 'STOCHF_k', 'STOCHF_d', 'STOCHRSI_k', 'STOCHRSI_d', 'TRIX_12', 'TRIX_24', 'TRIX_48', 'ULTOSC', 'WILLR_6', 'WILLR_12', 'WILLR_24', 'WILLR_48', 'AD', 'ADOSC', 'OBV', 'ATR_14', 'ATR_28', 'NATR_14', 'NATR_28', 'TRANGE', 'LINEARREG_SLOPE_5', 'LINEARREG_SLOPE_14', 'LINEARREG_SLOPE_28', 'TSF_5', 'TSF_10', 'TSF_20', 'TSF_40', 'VAR_5', 'VAR_10', 'VAR_20', 'VAR_40', 'TURN_RATE_LN', 'TURN_MAX_5', 'TURN_MAX_10', 'TURN_MAX_20', 'TURN_MAX_40', 'TURN_MIN_5', 'TURN_MIN_10', 'TURN_MIN_20', 'TURN_MIN_40', 'TURN_RATE_EMA_5', 'TURN_RATE_EMA_10', 'TURN_RATE_EMA_20', 'TURN_ROC_5', 'TURN_ROC_10', 'TURN_ROC_20', 'TURN_ROC_40', 'TURN_SLOPE_5', 'TURN_SLOPE_10', 'TURN_SLOPE_20', 'TURN_SLOPE_40', 'TURN_RSI_5', 'TURN_RSI_10', 'TURN_RSI_20', 'TURN_RSI_40', 'TURN_TSF_5', 'TURN_TSF_10', 'TURN_TSF_20', 'TURN_TSF_40', 'AMOUNT_LN', 'AMT_MAX_5', 'AMT_MAX_10', 'AMT_MAX_20', 'AMT_MAX_40', 'AMT_MIN_5', 'AMT_MIN_10', 'AMT_MIN_20', 'AMT_MIN_40', 'AMT_EMA_5', 'AMT_EMA_10', 'AMT_EMA_20', 'AMT_EMA_40', 'AMT_ROC_5', 'AMT_ROC_10', 'AMT_ROC_20', 'AMT_ROC_40', 'AMT_TRIX_5', 'AMT_TRIX_10', 'AMT_TRIX_20', 'AMT_TRIX_40', 'AMT_SLOPE_5', 'AMT_SLOPE_10', 'AMT_SLOPE_20', 'AMT_SLOPE_40', 'AMT_RSI_5', 'AMT_RSI_10', 'AMT_RSI_20', 'AMT_RSI_40', 'AMT_TSF_5', 'AMT_TSF_10', 'AMT_TSF_20', 'AMT_TSF_40', 'AMT_VAR_5', 'AMT_VAR_10', 'AMT_VAR_20', 'AMT_VAR_40']
                #print(new_features.info())
                # <class 'pandas.core.frame.DataFrame'>
                # DatetimeIndex: 2055 entries, 2016-08-15 to 2025-02-10
                # Columns: 140 entries, SIZE to AMT_VAR_40
                # dtypes: float64(140)
                # memory usage: 2.2 MB
                #print(new_features.tail())

                slice_features = self.generate_slice_features(code)
                #slice_features = slice_features[slice_features.index > last_date]
                #print(slice_features.info())
                # <class 'pandas.core.frame.DataFrame'>
                # DatetimeIndex: 4 entries, 2025-01-27 to 2025-02-07
                # Data columns (total 3 columns):
                #  #   Column              Non-Null Count  Dtype  
                # ---  ------              --------------  -----  
                #  0   DAILY_AMOUNT_RATIO  4 non-null      float64
                #  1   STR_FACTOR          4 non-null      object ———强制转化为float64
                #  2   RANK                4 non-null      float64
                # dtypes: float64(2), object(1)
                # memory usage: 128.0+ bytes
                # <class 'pandas.core.frame.DataFrame'>
                # DatetimeIndex: 2055 entries, 2016-08-15 to 2025-02-10
                # Data columns (total 3 columns):
                # #   Column              Non-Null Count  Dtype  
                # ---  ------              --------------  -----  
                # 0   DAILY_AMOUNT_RATIO  2055 non-null   float64
                # 1   STR_FACTOR          2043 non-null   float64
                # 2   RANK                2055 non-null   float64
                # dtypes: float64(3)
                # memory usage: 128.8 KB
                #print(slice_features.tail())

                rsrs_features = self.generate_rsrs_features(df=rsrs_df, stock_id=code)
                #rsrs_features = rsrs_features[rsrs_features.index > last_date]
                #print(rsrs_features.info())
                # <class 'pandas.core.frame.DataFrame'>
                # DatetimeIndex: 4 entries, 2025-01-27 to 2025-02-07
                # Data columns (total 4 columns):
                # #   Column       Non-Null Count  Dtype  
                # ---  ------       --------------  -----  
                # 0   base_RSRS    4 non-null      float64
                # 1   norm_RSRS    4 non-null      float64
                # 2   revise_RSRS  4 non-null      float64
                # 3   pos_RSRS     4 non-null      float64
                # dtypes: float64(4)
                # memory usage: 160.0 bytes
                # <class 'pandas.core.frame.DataFrame'>
                # DatetimeIndex: 2545 entries, 2014-07-17 to 2025-02-10
                # Data columns (total 4 columns):
                # #   Column       Non-Null Count  Dtype  
                # ---  ------       --------------  -----  
                # 0   base_RSRS    2526 non-null   float64
                # 1   norm_RSRS    2027 non-null   float64
                # 2   revise_RSRS  2027 non-null   float64
                # 3   pos_RSRS     2027 non-null   float64
                # dtypes: float64(4)
                # memory usage: 99.4 KB
                #print(rsrs_features.tail())

                input_df = df[df.index > last_date]

                if cpv_df is None:
                    # 列为空值
                    cpv_df = pd.DataFrame()
                    cpv_df["cpv"] = np.nan
                # 情况 2: 检查是否是 Pandas DataFrame 且为空
                elif isinstance(cpv_df, pd.DataFrame) and cpv_df.empty:
                    cpv_df["cpv"] = np.nan
                # 情况 3: 检查是否是 Pandas DataFrame 且不为空
                elif isinstance(cpv_df, pd.DataFrame) and not cpv_df.empty:
                    pass # do nothing
                else:
                    raise ValueError("Invalid state for cpv_df.")
                    
                # 拼接数据
                #对齐索引
                new_features_aligned = new_features.reindex(input_df.index)
                slice_features_aligned = slice_features.reindex(input_df.index)
                rsrs_features_aligned = rsrs_features.reindex(input_df.index)
                cpv_feature_aligned = cpv_df.reindex(input_df.index)

                updated_df = pd.concat([input_df, new_features_aligned, slice_features_aligned, rsrs_features_aligned, cpv_feature_aligned], axis=1)
                #print(updated_df.columns.to_list())
                #print(updated_df.head())
                #print(updated_df.info())
                # <class 'pandas.core.frame.DataFrame'>
                # DatetimeIndex: 4 entries, 2025-01-27 to 2025-02-07
                # Columns: 170 entries, open to pos_RSRS
                # dtypes: float64(170)
                # memory usage: 5.3 KB

                # 读取已有数据
                existing_df = pd.read_csv(output_path, index_col=0, parse_dates=True)
                #print(existing_df.info())

                # 合并新旧数据并按索引排序
                result = pd.concat([existing_df, updated_df[~updated_df.index.isin(existing_df.index)]], axis=0)
                # 保存结果
                result.to_csv(output_path)
                return result.columns.tolist()

            else:
                
                
                new_features = self.generate_single_stock_features(df)
                slice_features = self.generate_slice_features(code)
                rsrs_features = self.generate_rsrs_features(df=rsrs_df,stock_id=stock_id)
                input_df = df

                if cpv_df is None:
                    # 列为空值
                    cpv_df = pd.DataFrame()
                    cpv_df["cpv"] = np.nan
                # 情况 2: 检查是否是 Pandas DataFrame 且为空
                elif isinstance(cpv_df, pd.DataFrame) and cpv_df.empty: 
                    cpv_df["cpv"] = np.nan
                # 情况 3: 检查是否是 Pandas DataFrame 且不为空
                elif isinstance(cpv_df, pd.DataFrame) and not cpv_df.empty:
                    # print(cpv_df.info())
                    # <class 'pandas.core.frame.DataFrame'>
                    # DatetimeIndex: 3860 entries, 2008-01-02 to 2024-12-31
                    # Data columns (total 1 columns):
                    # #   Column  Non-Null Count  Dtype  
                    # ---  ------  --------------  -----  
                    # 0   cpv     3586 non-null   float64
                    # dtypes: float64(1)
                    # memory usage: 60.3 KB
                    pass # do nothing
                else:
                    raise ValueError("Invalid state for cpv_df.")
                
                # 检查bug
                if stock_id =="sz002457":
                    print("######new_features######")
                    print(new_features.info())       
                    print("######slice_features######")
                    print(slice_features.info())
                    print("######rsrs_features######")
                    print(rsrs_features.info())
                    print("######cpv_df######")
                    print(cpv_df.info())
                    print("-------------------")
                    print(f"index duplicate status: {new_features.index.has_duplicates} {slice_features.index.has_duplicates} {rsrs_features.index.has_duplicates} {cpv_df.index.has_duplicates}")
                    print(f"columns duplicate status: {new_features.columns.has_duplicates}  {slice_features.columns.has_duplicates} {rsrs_features.columns.has_duplicates} {cpv_df.columns.has_duplicates}")

                    # 查看重复的索引值
                    duplicates = cpv_df.index[cpv_df.index.duplicated()]
                    print(duplicates)

                # 合并特征
                #对齐索引
                new_features_aligned = new_features.reindex(input_df.index)
                slice_features_aligned = slice_features.reindex(input_df.index)
                rsrs_features_aligned = rsrs_features.reindex(input_df.index)
                cpv_feature_aligned = cpv_df.reindex(input_df.index)

                updated_df = pd.concat([input_df, new_features_aligned, slice_features_aligned, rsrs_features_aligned, cpv_feature_aligned], axis=1)

                # 保存结果
                updated_df.to_csv(output_path)
                return updated_df.columns.tolist()
        except Exception as e:
            print(f"Error processing {stock_id}: {e}")
            return []

    def process_directory_incremental(self, input_dir, output_dir, feature_meta_file=None):
        """增量处理整个目录的CSV文件"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print("Scanning files...")
        stock_data = {}
        stock_data_rsrs = {}
        update_files = []
        effective_count = 0
        empty_count = 0
        short_count = 0
        feature_names = set()

        # 1. 扫描文件并检查更新状态
        for file_path in Path(input_dir).glob('*.csv'):
            filename = file_path.name
            output_path = Path(output_dir) / filename
            stock_id = filename.rsplit('.', 1)[0]

            if any(code in filename for code in self.stock_codes_set):
                try:
                    ### 如果stock id不在stock_pool中，则跳过
                    if self.stock_pool is not None and stock_id not in self.stock_pool:
                        continue
                    
                    status = self._check_file_status(file_path, output_path)
                    # {
                    #     'needs_update': needs_update,
                    #     'last_date': last_date,
                    #     'df_length':df_length,
                    #     'input_df': df_global_cast,
                    #     'input_rsrs_df': df_rsrs_cast            
                    # }

                    if status['needs_update']:
                        if status['input_df'].empty:
                            empty_count += 1
                            continue

                        # 如果数据长度不足，则跳过
                        if status['df_length'] < self.minimum_data_length:
                            short_count += 1
                            continue

                        stock_data[stock_id] = status['input_df']
                        stock_data_rsrs[stock_id] = status['input_rsrs_df']
                        last_date = status['last_date']
                        update_files.append({
                            'stock_id': stock_id,
                            'last_date': last_date,
                            'output_path': output_path
                        })
                        effective_count += 1

                except Exception as e:
                    print(f"Error checking {filename}: {e}")

            if self.index_code in filename: # 如果是指数
                print(f"Reading index file: {file_path.name}")
                df = pd.read_csv(file_path, parse_dates=['date'])

                # 保存指数的dataframe
                self.index_df = df
        print(f"Found {len(update_files)} files to update")

        if not update_files:
            print("No files need to be updated.")
            return

        # 2. 计算横截面特征
        print("Calculating cross-sectional features...")
        self.pre_process_slice_features(stock_data)

        # 3. 并行处理需要更新的文件
        print("Processing updates...")
        if self.cpv_df is not None:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._process_stock_data_incremental)(
                    file_info['stock_id'],
                    stock_data[file_info['stock_id']],
                    stock_data_rsrs[file_info['stock_id']],
                    self.cpv_df[file_info['stock_id']] if file_info['stock_id'] in self.cpv_df else None,
                    file_info['output_path'],
                    file_info['last_date']
                )
                for file_info in tqdm(update_files, desc="Updating files")
            )
        else:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._process_stock_data_incremental)(
                    file_info['stock_id'],
                    stock_data[file_info['stock_id']],
                    stock_data_rsrs[file_info['stock_id']],
                    None,
                    file_info['output_path'],
                    file_info['last_date']
                )
                for file_info in tqdm(update_files, desc="Updating files")
            )
        # 保存缓存
        self.rsrs_feature_cls.save_cache()

        # 4. 更新特征名称
        for cols_new in results:
            feature_names.update(cols_new)
        self.output_feature_meta(file_path=feature_meta_file, feature_name_set=feature_names)

        print(f"Successfully processed {len(results)} files")

        # 5. 保存指数数据
        if not self.index_df.empty:
            output_path = Path(output_dir) / f"{self.index_code}.csv"
            self.index_df.to_csv(output_path)
        else:
            print("No index data found or index data is empty")

    def output_feature_meta(self, file_path: str, feature_name_set: set):
        """_summary_

        Args:
            file_path (str): _description_
            feature_name_set (set): _description_
        """
        # 生成feature dict
        feature_meta_dic = {}
        feature_meta_dic['fields'] = []
        feature_meta_dic['names'] = []
        feature_meta_dic['description'] = "version: v4.2, code time: 2025-02-012, scope: ta-lib/STR/RSRS, feature count: %d" % len(feature_name_set)

        for feature_name in sorted(feature_name_set):
            feature_meta_dic['fields'].append(f"${feature_name}")
            feature_meta_dic['names'].append(feature_name)

        ##将dict 输出到json文件中
        with open(file_path, 'w') as f:
            json.dump(feature_meta_dic, f, indent=4)

def __test__():
    working_folder = '/home/godlike/project/GoldSparrow/Day_Data' ## 本地测试
    #working_folder = '/root/autodl-tmp/GoldSparrow/Day_Data' ## 服务器测试

    basic_info_path = f"{working_folder}/qlib_data/basic_info.csv"
    in_folder = f"{working_folder}/test_raw"
    out_folder = f"{working_folder}/test_raw_ta"
    feature_meta_file = f"{working_folder}/feature_names.json"
    cpv_feature_config_path = "/home/godlike/project/GoldSparrow/Min_Data/config.json"
    #stock_pool_file = f"{working_folder}/qlib_data/instruments/csi300.txt"

    feature_generator = TALibFeatureExt(
        basic_info_path=basic_info_path,
        time_range=5,
        stock_pool_path=None,
        cpv_feature_config_path=cpv_feature_config_path,
        n_jobs=-1
    )
    # 使用增量更新方法
    feature_generator.process_directory_incremental(in_folder, out_folder, feature_meta_file)

if __name__ == '__main__':
    __test__()