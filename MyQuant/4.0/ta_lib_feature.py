"""_summary_

    Returns:
        _type_: _description_
"""
from pathlib import Path

import pandas as pd
import talib

class TALibFeature:
    def __init__(self):
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
            'TRANGE': {}
        }

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

        return pd.DataFrame(features, index=df.index)

    def process_directory(self, input_dir, output_dir):
        """处理整个目录的CSV文件"""
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 处理每个CSV文件
        for file_path in Path(input_dir).glob('*.csv'):
            try:
                # 读取原始数据
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)

                # 生成新特征
                new_features = self.generate_single_stock_features(df)

                # 合并特征
                result = pd.concat([df, new_features], axis=1)

                # 保存结果
                output_path = Path(output_dir) / file_path.name
                result.to_csv(output_path)
                print(f"Successfully processed {file_path.name}")

            except Exception as e:
                print(f"Error processing {file_path.name}: {str(e)}")


def __test__():
    # 测试特征生成器
    feature_generator = TALibFeature()
    in_folder = '/home/godlike/project/GoldSparrow/Day_Data/Day_data/test_raw'
    out_folder = '/home/godlike/project/GoldSparrow/Day_Data/Day_data/test_raw_ta'
    feature_generator.process_directory(in_folder, out_folder)

if __name__ == '__main__':
    __test__()