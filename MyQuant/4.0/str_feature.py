# title: str_feature.py
# updated: 2025.1.4
# change log:
#   - 计算凸显性因子STR
#   - ref: https://www.bilibili.com/video/BV1LJsve5Eso/
#   - 为了提升速度，使用qlib相关的数据结构
#
# 输入：
#   - qlib目录：需要将数据读取
#   - 原始的csv文件目录
#
# 输出：
#   - merged folder

# 目标：
#   1. 计算STR凸显性因子
#   2. 拼接原始csv文件，产出合并后的csv文件

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import qlib
from qlib.data import D


class STRFeature:
    """_summary_
    """
    def __init__(self, 
                 basic_info_path: str = None,
                 qlib_provider_uri: str = None):
        """_summary_

        Args:
            basic_info_path (str): _description_
        """

        self.use_qlib = False
        self.stock_codes_set = set()

        if  basic_info_path and Path(basic_info_path).is_file():
            self.basic_info_df = pd.read_csv(basic_info_path)
            # 去掉code列中的点号，并转换为小写
            self.basic_info_df['code'] = self.basic_info_df['code'].str.replace('.', '').str.lower()
            # 筛选出type为1的股票代码，并转换为集合以提高查找效率
            self.stock_codes_set = set(self.basic_info_df[self.basic_info_df['type'] == 1]['code'])

        if qlib_provider_uri:
            qlib.init(provider_uri = qlib_provider_uri)
            self.use_qlib = True

    def _calc_sigma(self, return_df: pd.DataFrame, theta=0.1):
        """_summary_
            计算凸显性系数

        Args:
            return_df (_type_): dataframe, 行索引为时间，列索引是股票代码
            theta (float, optional): 超参数，控制零收益率的影响
        """
        _median = return_df.median(axis=1)
        df_frac1 = return_df.sub(_median,axis=0).abs()
        df_frac2 = return_df.abs().add(_median.abs(),axis=0) + theta
        return df_frac1.div(df_frac2)

    def _calc_weight(self, sigma: pd.DataFrame, time_range=22, delta= 0.7):
        """_summary_
            计算凸显性权重
        Args:
            sigma (_type_): _description_
            time_rage (int, optional): _description_. Defaults to 22.
            delta (float, optional): _description_. Defaults to 0.7.
        """
        df = sigma.iloc[-time_range:,:]
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
             time_range = 22):
        """_summary_

        Args:
            weight (pd.DataFrame): _description_
        """
        return weight.rolling(time_range).cov(return_df).iloc[-1,:]
    def compute_with_qlib(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        if self.use_qlib:
            all_code = D.list_instruments(D.instruments("all"), as_list=True)
            returns = D.features(all_code, fields=["$close/Ref($close,1)-1"])
            returns = returns.unstack(level=0)["$close/Ref($close,1)-1"]
            
            time_range = 22
            sigma = self._calc_sigma(returns)
            weight = self._calc_weight(sigma, time_range)
            return self._STR_factor(weight,returns,time_range)
    
    def process_directory(self, input_dir, output_dir):
        """_summary_

        Args:
            input_dir (_type_): _description_
            output_dir (_type_): _description_
        """
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for file_path in Path(input_dir).glob('*.csv'):
            filename = file_path.name
            if any(code in filename for code in self.stock_codes_set):
                # 读取CSV文件
                df = pd.read_csv(file_path,parse_dates=['date'])

def __test__():
    # 测试特征生成器
    basic_info_path = '/home/godlike/project/GoldSparrow/Day_Data/Day_data/qlib_data/basic_info.csv'
    qlib_folder = '/home/godlike/project/GoldSparrow/Day_Data/Day_data/qlib_data'
    feature_generator = STRFeature(basic_info_path=basic_info_path, qlib_provider_uri=qlib_folder)
    in_folder = '/home/godlike/project/GoldSparrow/Day_Data/Day_data/test_raw'
    out_folder = '/home/godlike/project/GoldSparrow/Day_Data/Day_data/test_raw_ta'
    feature_generator.process_directory(in_folder, out_folder)

if __name__ == '__main__':
    __test__()