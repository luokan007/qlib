# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
# from re import I
import sys
# import copy
# import time
# import datetime
# import importlib
from abc import ABC
# import multiprocessing
from pathlib import Path
# from typing import Iterable

# import fire
# import requests
# import numpy as np
import pandas as pd
from loguru import logger
# from yahooquery import Ticker
# from dateutil.tz import tzlocal

# from qlib.tests.data import GetData
# from qlib.utils import code_to_fname, fname_to_code, exists_qlib_data
# from qlib.constant import REG_CN as REGION_CN

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

# from dump_bin import DumpDataUpdate
# from data_collector.base import BaseCollector, BaseNormalize, BaseRun, Normalize
import akshare as ak
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
INDEX_BENCH_URL = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.{index_code}&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&beg={begin}&end={end}"


        # self._source_dir = Path(source_dir).expanduser()
        # self._target_dir = Path(target_dir).expanduser()
        # self._target_dir.mkdir(parents=True, exist_ok=True)

# 东财接口
# 东财个股行情下载与保存
# 最终csv包含字段 date,open,close,high,low,volume,amount,turnover,adjclose,symbol                 
COLUMNS_RENAME={'日期':'date', '开盘':'open', '收盘':'close', '最高':'high', '最低':'low', 
                '成交量':'volume', '成交额':'amount', '换手率':'turnover'} # 列名对照表
COLUMNS = ["open", "close", "high", "low", "volume"] # qtb修订，列出需要复权的字段. 根据不同数据源进行定制。


class MyNormalize(abc.ABC):
    def __init__(self, 
        date_field_name: str = "date", 
        symbol_field_name: str = "symbol",
        source_dir: str=None,
        target_dir: str=None,        
        max_workers: int = 16):

        if not (source_dir and target_dir):
            raise ValueError("source_dir and target_dir cannot be None")
        self._source_dir = Path(source_dir).expanduser()
        self._target_dir = Path(target_dir).expanduser()
        self._target_dir.mkdir(parents=True, exist_ok=True)
        self._date_field_name = date_field_name
        self._symbol_field_name = symbol_field_name
        # self._end_date = kwargs.get("end_date", None)
        self._max_workers = max_workers

    # @abc.abstractmethod
    # def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
    #     # normalize
    #     raise NotImplementedError("")

    # @abc.abstractmethod
    # def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
    #     """Get benchmark calendar"""
    #     raise NotImplementedError("")


    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
      
        df = df.copy()
        df.set_index(self._date_field_name, inplace=True)
        if "adjclose" in df:
            # 计算复权因子
            df["factor"] = df["adjclose"] / df["close"]
            df["factor"] = df["factor"].fillna(method="ffill")
        else:
            df["factor"] = 1
        for _col in COLUMNS:
            if _col not in df.columns:
                continue
            if _col in ["volume", "outstanding_share"]: # qtb 修订，增加outstanding_share
                # 复权成交量，流通股数
                df[_col] = df[_col] / df["factor"]
            else:
                # 复权其它字段
                df[_col] = df[_col] * df["factor"]
        df.index.names = [self._date_field_name]
        return df.reset_index()	

    def executor(self, file_path: Path):
 
        file_path = Path(file_path)
        df = pd.read_csv(file_path, dtype = {'symbol':str}) # qtb 修订

        

        
        df = self.adjusted_price(df)


        if df is not None and not df.empty:   
            
            df.to_csv(self._target_dir.joinpath(file_path.name), index=False)

    def normalize(self):
        logger.info("normalize data......")

        with ProcessPoolExecutor(max_workers=self._max_workers) as worker:
            file_list = list(self._source_dir.glob("*.csv"))
            with tqdm(total=len(file_list)) as p_bar:
                for _ in worker.map(self.executor, file_list):
                    p_bar.update()

class DownloadData(abc.ABC):
    def __init__(self, 
        # date_field_name: str = "date", 
        # symbol_field_name: str = "symbol",
        source_dir: str=None,
        # target_dir: str=None,        
        
        # period: str ='daily', 
        start_date: str=None, 
        end_date: str=None, 
        adjust: str='hfq',
        all_codes: list = None,
        max_workers: int = 16,
        ):

        if not (source_dir):
            raise ValueError("source_dir and target_dir cannot be None")
        self._source_dir = Path(source_dir).expanduser()
        
        # self._target_dir = Path(target_dir).expanduser()
        # self._target_dir.mkdir(parents=True, exist_ok=True)
        # self._date_field_name = date_field_name
        # self._symbol_field_name = symbol_field_name

        # self._period = period
        self._start_date = start_date
        self._end_date = end_date
        self._adjust = adjust
        self._all_codes = all_codes
        self._max_workers = max_workers

        



    def download_data(self, code):
        '''传入股票代码code必须无前缀，形如600000，下面代码会自动加cn前缀'''
        symbol = 'cn' + code # 代码加前缀cn  
        try:
            # 原始日线数据
            kline_df_raw = ak.stock_zh_a_hist(symbol=code, start_date=self._start_date, end_date=self._end_date, adjust="") 
            if kline_df_raw is None or kline_df_raw.empty:
                print('code',code , '未返回数据')
                return        
        
            # 后复权日线数据 
            kline_df_adj = ak.stock_zh_a_hist(symbol=code, start_date=self._start_date, end_date=self._end_date, 
                                            adjust=self._adjust)[['日期','收盘']].rename(columns={'收盘':'adjclose'}) 
            if kline_df_adj is None or kline_df_adj.empty:
                print('code2',code , '未返回数据')
                return
                
            # 合并原始价格数据和复权价格
            kline_df_all = pd.merge(kline_df_raw, kline_df_adj, how='left', on='日期')[['日期','开盘', '收盘', 
                            '最高', '最低', '成交量', '成交额', '换手率', 'adjclose']].rename(columns=COLUMNS_RENAME)
        except Exception as e:
            print(e)
            return 
        
        kline_df_all['symbol'] = symbol # 创建symbol列  
        instrument_path = self._source_dir.joinpath(f"{symbol}.csv") 

        kline_df_all.to_csv(instrument_path, index=False) # 保存到csv

        
    
    def download(self):
        with ProcessPoolExecutor(max_workers=self._max_workers) as worker:         
            with tqdm(total=len(self._all_codes)) as p_bar:
                for _ in worker.map(self.download_data, self._all_codes):
                    p_bar.update()
        
        # # 并发下载
        # for i in range(5):
        #     if len(all_codes[i*1000:(i+1)*1000]) == 0:
        #         break
            
        #     # 下载数据大约需要10分钟
        #     print(f'round {i} begin ')
        #     with ThreadPoolExecutor(max_workers=20) as executor:   
        #         _ = [executor.submit(download_data, code,  period='daily', start_date=start_date, end_date=end_date, adjust='hfq') for code in all_codes[i*1000:(i+1)*1000]] 
                
        #     print(f'round {i} finished')

        #     if len(all_codes[i*1000:(i+1)*1000]) < 1000:
        #         break
        #     else:
        #         print('sleeping')
        #         sleep(60)
        #         print('sleeping finished')

    # 东财指数行情下载与保存
    def download_index_data(self, code): 
        '''参数code必须无前缀'''
        symbol = 'ix'+ code 
        index_df = ak.index_zh_a_hist(symbol=code, period="daily", start_date=self._start_date, end_date=self._end_date).drop(columns=['振幅','涨跌幅','涨跌额']).rename(columns=COLUMNS_RENAME)    
        index_df['adjclose'] = index_df['close'] # 增加复权价列      
        index_df['symbol'] = symbol # 指数加前缀ix 
        instrument_path = self._source_dir.joinpath(f"{symbol}.csv")                             
        index_df.to_csv(instrument_path, index=False) # 保存到csv


