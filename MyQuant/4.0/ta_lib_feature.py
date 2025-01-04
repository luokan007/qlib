"""_summary_

    Returns:
        _type_: _description_
"""
from pathlib import Path
import numpy as np
import pandas as pd
import talib
from tqdm import tqdm

class TALibFeature:
    def __init__(self, basic_info_path):
        """初始化特征生成器"""
        # 定义要生成的技术指标及其参数
        self.feature_functions = {
            'EMA': {'periods': [5, 10, 20]},
            'SAR': {},
            
            'RSI': {'periods': [6, 12, 24]},
            'ADX': {'timeperiod': [14,28]},
            'BOP': {},
            'CCI': {'timeperiod': [14,28]},
            'MACD': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9},
            'MACDEXT': {},
            'MOM': {'timeperiod': [6, 12, 24, 48]},
            'ULTOSC': {},
            'WILLR': {'timeperiod': [6, 12, 24, 48]},
            
            'AD': {},
            'ADOSC': {'fastperiod': 3, 'slowperiod': 10},
            'OBV': {},
            
            'ATR': {'timeperiod': [14, 28]},
            'NATR': {'timeperiod': [14, 28]},
            'TRANGE': {},
            
            'TURN_RATE': {'periods': [5, 10, 20]},
            'TURN_RATE_MIX':{},
            
            'BETA': {}
        }
        
        self.amount_df = pd.DataFrame(columns=['date', 'amount'])
        self.index_code = "sh000300"
        self.index_df = pd.DataFrame(columns=['date', 'pctChg'])
        
        self.basic_info_df = pd.read_csv(basic_info_path)
        
        # 去掉code列中的点号，并转换为小写
        self.basic_info_df['code'] = self.basic_info_df['code'].str.replace('.', '').str.lower()
        # 筛选出type为1的股票代码，并转换为集合以提高查找效率
        self.stock_codes_set = set(self.basic_info_df[self.basic_info_df['type'] == 1]['code'])
        

    def add_to_total_amount(self, df):
        target_df = df[['date', 'amount']]
        self.amount_df = pd.concat([self.amount_df, target_df], ignore_index=True)

    def get_total_amount(self):
        # 按日期分组并对amount列求和
        self.amount_df = self.amount_df.groupby('date')['amount'].sum().reset_index()
        # 设置'date'列为索引
        self.amount_df.set_index('date', inplace=True)
        self.amount_df.sort_index(inplace=True)
        self.amount_df["amount"] = np.log(self.amount_df['amount'])
    
    def generate_slice_features(self, df):
        """新增基于横截面的特征
        - 成交量占比

        Args:
            df (pd.DataFrame): 包含 'date' 和 'amount' 列的 DataFrame，其中 'date' 是索引

        Returns:
            pd.DataFrame: 包含 'volume_ratio' 列的新 DataFrame
        """
        # 确保 'amount' 列已经取对数
        if not df.empty:
            df['log_amount'] = np.log(df['amount'] + 1e-5)

            # 获取整体的对数金额（假设 self.amount_df 已经包含整体的对数金额数据）
            overall_log_amount = self.amount_df.loc[df.index, 'amount']

            # 检查是否有缺失值
            if df['log_amount'].isnull().any() or overall_log_amount.isnull().any():
                print("Warning: NaN values found in log_amount or overall_log_amount")
                return pd.DataFrame(index=df.index, columns=['volume_ratio'])

            # 计算成交量占比
            df['volume_ratio'] = df['log_amount'] - overall_log_amount

            # 返回包含新特征的 DataFrame
            return df[['volume_ratio']]

        # 如果 df 为空，返回一个空的 DataFrame
        return pd.DataFrame(columns=['volume_ratio'])

    def generate_single_stock_features(self, df):
        """_summary_
            新增更多基于ta-lib的特征
            - Overlap Studies(重叠指标)
                - EMA,Exponential Moving Average （指数移动平均线）
                        //- BBANDS,Bollinger Bands （布林带）
                - SAR,Parabolic SAR （抛物线转向）
            - Momentum Indicators(动量指标)
                - RSI,Relative Strength Index （相对强弱指标）
                - ADX, Average Directional Movement Index
                - BOP, Balance Of Power
                - CCI, Commodity Channel Index
                - MACD, Moving Average Convergence/Divergence
                - MACDEXT, MACD with controllable MA type
                - MOM,Momentum
                - ULTOSC,Ultimate Oscillator
                - WILLR,Williams' %R
            - Volume Indicators(成交量指标)
                - AD, Chaikin A/D Line
                - ADOSC, Chaikin A/D Oscillator
                - OBV, On Balance Volume
            - Volatility Indicators(波动率指标)
                - ATR, Average True Range
                - NATR, Normalized Average True Range
                - TRANGE, True Range
            - Turnover Rate 换手率相关指标
                - EMA
                - MIX
            - beta 指标
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


        # EMA,Exponential Moving Average （指数移动平均线）
        for period in self.feature_functions['EMA']['periods']:
            features[f'EMA_{period}'] = talib.EMA(df[close_col], timeperiod=period)

        # SAR,Parabolic SAR （抛物线转向）
        features['SAR'] = talib.SAR(df[high_col], df[low_col])

        # RSI,Relative Strength Index （相对强弱指标）
        for period in self.feature_functions['RSI']['periods']:
            features[f'RSI_{period}'] = talib.RSI(df[close_col], timeperiod=period)

        # ADX, Average Directional Movement Index
        for period in self.feature_functions['ADX']['timeperiod']:
            features[f'ADX_{period}'] = talib.ADX(df[high_col], df[low_col], df[close_col], timeperiod=period)

        # BOP, Balance Of Power
        features['BOP'] = talib.BOP(df[open_col], df[high_col], df[low_col], df[close_col])

        # CCI, Commodity Channel Index
        for period in self.feature_functions['CCI']['timeperiod']:
            features[f'CCI_{period}'] = talib.CCI(df[high_col], df[low_col], df[close_col], timeperiod=period)
        
        # MACD, Moving Average Convergence/Divergence
        macd, signal, hist = talib.MACD(df[close_col],
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
            features[f'MOM_{period}'] = talib.MOM(df[close_col], timeperiod=period)

        # ULTOSC,Ultimate Oscillator
        features['ULTOSC'] = talib.ULTOSC(df[high_col], df[low_col], df[close_col])
        
        # WILLR,Williams' %R
        for period in self.feature_functions['WILLR']['timeperiod']:
            features[f'WILLR_{period}'] = talib.WILLR(df[high_col], df[low_col], df[close_col], timeperiod=period)
        
        # AD, Chaikin A/D Line
        features['AD'] = talib.AD(df[high_col], df[low_col], df[close_col], df[volume_col])
        
        # ADOSC, Chaikin A/D Oscillator
        features['ADOSC'] = talib.ADOSC(df[high_col], df[low_col], df[close_col], df[volume_col], 
                                        fastperiod=self.feature_functions['ADOSC']['fastperiod'], 
                                        slowperiod=self.feature_functions['ADOSC']['slowperiod'])
        # 生成OBV
        features['OBV'] = talib.OBV(df[close_col], df[volume_col])
        
        # ATR, Average True Range
        for period in self.feature_functions['ATR']['timeperiod']:
            features[f'ATR_{period}'] = talib.ATR(df[high_col], df[low_col], df[close_col], timeperiod=period)
        
        # NATR, Normalized Average True Range
        for period in self.feature_functions['NATR']['timeperiod']:
            features[f'NATR_{period}'] = talib.NATR(df[high_col], df[low_col], df[close_col], timeperiod=period)
        
        # TRANGE, True Range
        features['TRANGE'] = talib.TRANGE(df[high_col], df[low_col], df[close_col])
        
        # Turnover Rate 换手率相关指标
        for period in self.feature_functions['TURN_RATE']['periods']:
            features[f'TURN_RATE_{period}'] = talib.EMA(df[turn_col], timeperiod=period)
        
        ## Turnover Rate MIX, 将日、周、月换手率的加权平均值作为新特征
        features['TURN_RATE_MIX'] = df[turn_col]*0.35 + features['TURN_RATE_5']*0.35 + features['TURN_RATE_20']*0.3
        
        ## beta 指标,
        comm_dates = df.index
        aligned_index_df = self.index_df.loc[comm_dates]
        aligned_index_df.fillna(0, inplace=True)
        features['BETA'] = talib.BETA(df[pct_chg_col], aligned_index_df[pct_chg_col], timeperiod=40)

        return pd.DataFrame(features, index=df.index)

    def process_directory(self, input_dir, output_dir):
        """处理整个目录的CSV文件"""
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print("Processing files for 1st round...")
        # 处理每个CSV文件，第一轮遍历，计算总成交量，获取指数数据
        for file_path in Path(input_dir).glob('*.csv'):
            filename = file_path.name
            if any(code in filename for code in self.stock_codes_set):
                # 读取CSV文件
                df = pd.read_csv(file_path,parse_dates=['date'])
                
                # 提取需要的列
                df = df[['date', 'amount']]
                
                # 将当前文件的数据追加到总的结果中
                self.add_to_total_amount(df)
        
            # 检查文件名是否包含 index_code: sh000300
            if self.index_code in filename:
                # 读取CSV文件
                df = pd.read_csv(file_path, parse_dates=['date'])
                
                # 提取需要的列
                self.index_df = df[['date', 'pctChg']]
                self.index_df.set_index('date', inplace=True)
        
        ##计算总成交量
        print("Calculating total amount...")
        self.get_total_amount()
        print("Processing files for 2nd round...")
         # 获取所有CSV文件列表
        files = list(Path(input_dir).glob('*.csv'))
        # 使用tqdm创建进度条
        for file_path in tqdm(files, desc="Processing stocks", unit="file"):
            try:
                # 读取原始数据
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                #print(f"Processing {file_path.name}...")

                if df.empty:
                    print(f"Empty file: {file_path.name}")
                    continue

                # 如果是股票，则生成新特征
                if any(code in file_path.name for code in self.stock_codes_set):
                    new_features = self.generate_single_stock_features(df)
                    slice_features = self.generate_slice_features(df)

                    # 合并特征
                    result = pd.concat([df, new_features], axis=1)
                    result = pd.concat([result, slice_features], axis=1)
                else:
                    result = df

                # 保存结果
                output_path = Path(output_dir) / file_path.name
                result.to_csv(output_path)
                #print(f"Successfully processed {file_path.name}")

            except Exception as e:
                print(f"Error processing {file_path.name}: {str(e)}")


def __test__():
    # 测试特征生成器
    basic_info_path = '/home/godlike/project/GoldSparrow/Day_Data/Day_data/qlib_data/basic_info.csv'
    feature_generator = TALibFeature(basic_info_path=basic_info_path)
    in_folder = '/home/godlike/project/GoldSparrow/Day_Data/Day_data/test_raw'
    out_folder = '/home/godlike/project/GoldSparrow/Day_Data/Day_data/test_raw_ta'
    feature_generator.process_directory(in_folder, out_folder)

if __name__ == '__main__':
    __test__()