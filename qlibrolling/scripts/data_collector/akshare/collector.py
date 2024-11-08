# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
from re import I
import sys
import copy
import time
import datetime
import importlib
from abc import ABC
import multiprocessing
from pathlib import Path
from typing import Iterable

import fire
import requests
import numpy as np
import pandas as pd
from loguru import logger
from yahooquery import Ticker
from dateutil.tz import tzlocal

from qlib.tests.data import GetData
from qlib.utils import code_to_fname, fname_to_code, exists_qlib_data
from qlib.constant import REG_CN as REGION_CN

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from dump_bin import DumpDataUpdate
from data_collector.base import BaseCollector, BaseNormalize, BaseRun, Normalize
import akshare as ak
INDEX_BENCH_URL = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.{index_code}&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&beg={begin}&end={end}"


# 东财接口
# 东财个股行情下载与保存
# 最终csv包含字段 date,open,close,high,low,volume,amount,turnover,adjclose,symbol                 
COLUMNS_RENAME={'日期':'date', '开盘':'open', '收盘':'close', '最高':'high', '最低':'low', 
                '成交量':'volume', '成交额':'amount', '换手率':'turnover'} # 列名对照表

def download_data(code, period='daily',start_date= "19700101", end_date="20500101",adjust='hfq'):
    '''传入股票代码code必须无前缀，形如600000，下面代码会自动加cn前缀'''
    symbol = 'cn' + code # 代码加前缀cn  
    try:
        # 原始日线数据
        kline_df_raw = ak.stock_zh_a_hist(symbol=code, start_date=start_date, end_date=end_date, adjust="") 
        if kline_df_raw is None or kline_df_raw.empty:
            print('code',code , '未返回数据')
            return        
    
        # 后复权日线数据 
        kline_df_adj = ak.stock_zh_a_hist(symbol=code, start_date=start_date, end_date=end_date, 
                                           adjust=adjust)[['日期','收盘']].rename(columns={'收盘':'adjclose'}) 
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
    kline_df_all.to_csv(f'.\\qlib_data\\ak_cn_data\\csv\\{symbol}.csv', index=False) # 保存到csv

# 东财指数行情下载与保存
def download_index_data(code, period="daily",start_date= "19700101", end_date="20500101"): 
    '''参数code必须无前缀'''
    symbol = 'ix'+ code 
    index_df = ak.index_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date).drop(columns=['振幅','涨跌幅','涨跌额']).rename(columns=COLUMNS_RENAME)    
    index_df['adjclose'] = index_df['close'] # 增加复权价列      
    index_df['symbol'] = symbol # 指数加前缀ix                                      
    index_df.to_csv(f'.\\qlib_data\\ak_cn_data\\csv\\{symbol}.csv', index=False) # 保存到csv

class AkshareNormalize(BaseNormalize):
    COLUMNS = ["open", "close", "high", "low", "volume"] # qtb修订，列出需要复权的字段. 根据不同数据源进行定制。
    
    DAILY_FORMAT = "%Y-%m-%d"

    @staticmethod
    def calc_change(df: pd.DataFrame, last_close: float) -> pd.Series:
        df = df.copy()
        _tmp_series = df["close"].fillna(method="ffill")
        _tmp_shift_series = _tmp_series.shift(1)
        if last_close is not None:
            _tmp_shift_series.iloc[0] = float(last_close)
        change_series = _tmp_series / _tmp_shift_series - 1
        return change_series

    @staticmethod
    def normalize_akshare(
        df: pd.DataFrame,
        calendar_list: list = None,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        last_close: float = None,
    ):
        if df.empty:
            return df
        symbol = df.loc[df[symbol_field_name].first_valid_index(), symbol_field_name]
        columns = copy.deepcopy(AkshareNormalize.COLUMNS)
        df = df.copy()
        df.set_index(date_field_name, inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep="first")]
        if calendar_list is not None:
            df = df.reindex(
                pd.DataFrame(index=calendar_list)
                .loc[
                    pd.Timestamp(df.index.min()).date() : pd.Timestamp(df.index.max()).date()
                    + pd.Timedelta(hours=23, minutes=59)
                ]
                .index
            )
        df.sort_index(inplace=True)
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), list(set(df.columns) - {symbol_field_name})] = np.nan # qtb 修改了

        change_series = AkshareNormalize.calc_change(df, last_close)
        # NOTE: The data obtained by Yahoo finance sometimes has exceptions
        # WARNING: If it is normal for a `symbol(exchange)` to differ by a factor of *89* to *111* for consecutive trading days,
        # WARNING: the logic in the following line needs to be modified
        _count = 0
        while True:
            # NOTE: may appear unusual for many days in a row
            change_series = AkshareNormalize.calc_change(df, last_close)
            _mask = (change_series >= 89) & (change_series <= 111)
            if not _mask.any():
                break
            _tmp_cols = ["high", "close", "low", "open", "adjclose"]
            df.loc[_mask, _tmp_cols] = df.loc[_mask, _tmp_cols] / 100 # 如果价格变化在89到111倍之间，所有价格缩小100倍
            _count += 1
            if _count >= 10: 
                # 如果价格变化大的记录太多，则报警
                _symbol = df.loc[df[symbol_field_name].first_valid_index()]["symbol"]
                logger.warning(
                    f"{_symbol} `change` is abnormal for {_count} consecutive days, please check the specific data file carefully"
                )

        # df["change"] = AkshareNormalize.calc_change(df, last_close)
        # columns += ["change"] # 增加收盘价相对昨日变化率列. 不要这个，否则以后新增数据的第一条记录本字段都是nan

        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), columns] = np.nan # 成交量不正常的记录，其它列设为nan

        df[symbol_field_name] = symbol
        df.index.names = [date_field_name]
        return df.reset_index()
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # 数据校验：设置日期索引，去重复日期。成交量小于等于0或isnan的记录，所有字段设为np.nan (symbol列不管)
		# 修订价格变化过大的记录
        df = self.normalize_akshare(df, self._calendar_list, self._date_field_name, self._symbol_field_name)
        
        # adjusted price 生成复权数据
        df = self.adjusted_price(df)
        return df

    @abc.abstractmethod
    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """adjusted price"""
        raise NotImplementedError("rewrite adjusted_price")


class AkshareNormalize1d(AkshareNormalize, ABC):
    DAILY_FORMAT = "%Y-%m-%d"

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
        for _col in self.COLUMNS:
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

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # 设置日期索引，去重复日期。成交量小于等于0或isnan的记录，所有字段设为np.nan (symbol列不管)
		# 修订价格变化过大的记录. 生成复权数据
        df = super(AkshareNormalize1d, self).normalize(df) 

        # df = self._manual_adj_data(df) # 用最早的close标准化所有数据 
        return df

    def _get_first_close(self, df: pd.DataFrame) -> float:
        """get first close value

        Notes
        -----
            For incremental updates(append) to Yahoo 1D data, user need to use a close that is not 0 on the first trading day of the existing data
        """
        df = df.loc[df["close"].first_valid_index() :]
        _close = df["close"].iloc[0]
        return _close

    def _manual_adj_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # 价格除以第一天的close标准化，成交量乘以close标准化
        """manual adjust data: All fields (except change) are standardized according to the close of the first day"""
        if df.empty:
            return df
        df = df.copy()
        df.sort_values(self._date_field_name, inplace=True)
        df = df.set_index(self._date_field_name)
        _close = self._get_first_close(df) # 第一天的close，已复权
        for _col in df.columns:
            # NOTE: retain original adjclose, required for incremental updates
            if _col in [self._symbol_field_name, "adjclose", "change"]:
                continue
            if _col in ["volume", "outstanding_share"]: # qtb 修订. 成交量，流通股标准化，注意是乘以_close
                df[_col] = df[_col] * _close
            elif  _col=='turnover':
                df[_col] = df[_col] * 100 # 换手率乘以100以便其数量级与其它指标相称，利于训练 
            else: # 其它字段open high low close标准化，除以_close. 	 
                df[_col] = df[_col] / _close # 
        return df.reset_index()






class AkshareNormalizeCN:
    # 从网址取得日历列表SZSE_CALENDAR_URL = "http://www.szse.cn/api/report/exchange/onepersistenthour/monthList?month={month}&random={random}"
    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        # TODO: from MSN
        # return get_calendar_list("ALL")
        pass


class AkshareNormalizeCN1d(AkshareNormalizeCN, AkshareNormalize1d):
    pass



class Run(BaseRun):
    def __init__(self, source_dir=None, normalize_dir=None, max_workers=1, interval="1d", region=REGION_CN):
        """

        Parameters
        ----------
        source_dir: str
            The directory where the raw data collected from the Internet is saved, default "Path(__file__).parent/source"
        normalize_dir: str
            Directory for normalize data, default "Path(__file__).parent/normalize"
        max_workers: int
            Concurrent number, default is 1; when collecting data, it is recommended that max_workers be set to 1
        interval: str
            freq, value from [1min, 1d], default 1d
        region: str
            region, value from ["CN", "US", "BR"], default "CN"
        """
        super().__init__(source_dir, normalize_dir, max_workers, interval)
        self.region = region

    @property
    def collector_class_name(self):
        return f"AkshareCollector{self.region.upper()}{self.interval}"

    @property
    def normalize_class_name(self):
        return f"AkshareNormalize{self.region.upper()}{self.interval}"

    @property
    def default_base_dir(self) -> [Path, str]:
        return CUR_DIR

    # def download_data(
    #     self,
    #     max_collector_count=2,
    #     delay=0.5,
    #     start=None,
    #     end=None,
    #     check_data_length=None,
    #     limit_nums=None,
    # ):
    #     """download data from Internet

    #     Parameters
    #     ----------
    #     max_collector_count: int
    #         default 2
    #     delay: float
    #         time.sleep(delay), default 0.5
    #     start: str
    #         start datetime, default "2000-01-01"; closed interval(including start)
    #     end: str
    #         end datetime, default ``pd.Timestamp(datetime.datetime.now() + pd.Timedelta(days=1))``; open interval(excluding end)
    #     check_data_length: int
    #         check data length, if not None and greater than 0, each symbol will be considered complete if its data length is greater than or equal to this value, otherwise it will be fetched again, the maximum number of fetches being (max_collector_count). By default None.
    #     limit_nums: int
    #         using for debug, by default None

    #     Notes
    #     -----
    #         check_data_length, example:
    #             daily, one year: 252 // 4
    #             us 1min, a week: 6.5 * 60 * 5
    #             cn 1min, a week: 4 * 60 * 5

    #     Examples
    #     ---------
    #         # get daily data
    #         $ python collector.py download_data --source_dir ~/.qlib/stock_data/source --region CN --start 2020-11-01 --end 2020-11-10 --delay 0.1 --interval 1d
    #         # get 1m data
    #         $ python collector.py download_data --source_dir ~/.qlib/stock_data/source --region CN --start 2020-11-01 --end 2020-11-10 --delay 0.1 --interval 1m
    #     """
    #     super(Run, self).download_data(max_collector_count, delay, start, end, check_data_length, limit_nums)

    def normalize_data(
        self,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        end_date: str = None,
        qlib_data_1d_dir: str = None,
    ):
        """normalize data

        Parameters
        ----------
        date_field_name: str
            date field name, default date
        symbol_field_name: str
            symbol field name, default symbol
        end_date: str
            if not None, normalize the last date saved (including end_date); if None, it will ignore this parameter; by default None
        qlib_data_1d_dir: str
            if interval==1min, qlib_data_1d_dir cannot be None, normalize 1min needs to use 1d data;

                qlib_data_1d can be obtained like this:
                    $ python scripts/get_data.py qlib_data --target_dir <qlib_data_1d_dir> --interval 1d
                    $ python scripts/data_collector/yahoo/collector.py update_data_to_bin --qlib_data_1d_dir <qlib_data_1d_dir> --trading_date 2021-06-01
                or:
                    download 1d data, reference: https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#1d-from-yahoo

        Examples
        ---------
            $ python collector.py normalize_data --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize --region cn --interval 1d
            $ python collector.py normalize_data --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data --source_dir ~/.qlib/stock_data/source_cn_1min --normalize_dir ~/.qlib/stock_data/normalize_cn_1min --region CN --interval 1min
        """
        if self.interval.lower() == "1min":
            if qlib_data_1d_dir is None or not Path(qlib_data_1d_dir).expanduser().exists():
                raise ValueError(
                    "If normalize 1min, the qlib_data_1d_dir parameter must be set: --qlib_data_1d_dir <user qlib 1d data >, Reference: https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#automatic-update-of-daily-frequency-datafrom-yahoo-finance"
                )
        super(Run, self).normalize_data(
            date_field_name, symbol_field_name, end_date=end_date, qlib_data_1d_dir=qlib_data_1d_dir
        )

    # def normalize_data_1d_extend(
    #     self, old_qlib_data_dir, date_field_name: str = "date", symbol_field_name: str = "symbol"
    # ):
    #     """normalize data extend; extending yahoo qlib data(from: https://github.com/microsoft/qlib/tree/main/scripts#download-cn-data)

    #     Notes
    #     -----
    #         Steps to extend yahoo qlib data:

    #             1. download qlib data: https://github.com/microsoft/qlib/tree/main/scripts#download-cn-data; save to <dir1>

    #             2. collector source data: https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#collector-data; save to <dir2>

    #             3. normalize new source data(from step 2): python scripts/data_collector/yahoo/collector.py normalize_data_1d_extend --old_qlib_dir <dir1> --source_dir <dir2> --normalize_dir <dir3> --region CN --interval 1d

    #             4. dump data: python scripts/dump_bin.py dump_update --csv_path <dir3> --qlib_dir <dir1> --freq day --date_field_name date --symbol_field_name symbol --exclude_fields symbol,date

    #             5. update instrument(eg. csi300): python python scripts/data_collector/cn_index/collector.py --index_name CSI300 --qlib_dir <dir1> --method parse_instruments

    #     Parameters
    #     ----------
    #     old_qlib_data_dir: str
    #         the qlib data to be updated for yahoo, usually from: https://github.com/microsoft/qlib/tree/main/scripts#download-cn-data
    #     date_field_name: str
    #         date field name, default date
    #     symbol_field_name: str
    #         symbol field name, default symbol

    #     Examples
    #     ---------
    #         $ python collector.py normalize_data_1d_extend --old_qlib_dir ~/.qlib/qlib_data/cn_data --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize --region CN --interval 1d
    #     """
    #     _class = getattr(self._cur_module, f"{self.normalize_class_name}Extend")
    #     yc = Normalize(
    #         source_dir=self.source_dir,
    #         target_dir=self.normalize_dir,
    #         normalize_class=_class,
    #         max_workers=self.max_workers,
    #         date_field_name=date_field_name,
    #         symbol_field_name=symbol_field_name,
    #         old_qlib_data_dir=old_qlib_data_dir,
    #     )
    #     yc.normalize()

    # def download_today_data(
    #     self,
    #     max_collector_count=2,
    #     delay=0.5,
    #     check_data_length=None,
    #     limit_nums=None,
    # ):
    #     """download today data from Internet

    #     Parameters
    #     ----------
    #     max_collector_count: int
    #         default 2
    #     delay: float
    #         time.sleep(delay), default 0.5
    #     check_data_length: int
    #         check data length, if not None and greater than 0, each symbol will be considered complete if its data length is greater than or equal to this value, otherwise it will be fetched again, the maximum number of fetches being (max_collector_count). By default None.
    #     limit_nums: int
    #         using for debug, by default None

    #     Notes
    #     -----
    #         Download today's data:
    #             start_time = datetime.datetime.now().date(); closed interval(including start)
    #             end_time = pd.Timestamp(start_time + pd.Timedelta(days=1)).date(); open interval(excluding end)

    #         check_data_length, example:
    #             daily, one year: 252 // 4
    #             us 1min, a week: 6.5 * 60 * 5
    #             cn 1min, a week: 4 * 60 * 5

    #     Examples
    #     ---------
    #         # get daily data
    #         $ python collector.py download_today_data --source_dir ~/.qlib/stock_data/source --region CN --delay 0.1 --interval 1d
    #         # get 1m data
    #         $ python collector.py download_today_data --source_dir ~/.qlib/stock_data/source --region CN --delay 0.1 --interval 1m
    #     """
    #     start = datetime.datetime.now().date()
    #     end = pd.Timestamp(start + pd.Timedelta(days=1)).date()
    #     self.download_data(
    #         max_collector_count,
    #         delay,
    #         start.strftime("%Y-%m-%d"),
    #         end.strftime("%Y-%m-%d"),
    #         check_data_length,
    #         limit_nums,
    #     )

    # def update_data_to_bin(
    #     self,
    #     qlib_data_1d_dir: str,
    #     trading_date: str = None,
    #     end_date: str = None,
    #     check_data_length: int = None,
    #     delay: float = 1,
    # ):
    #     """update yahoo data to bin

    #     Parameters
    #     ----------
    #     qlib_data_1d_dir: str
    #         the qlib data to be updated for yahoo, usually from: https://github.com/microsoft/qlib/tree/main/scripts#download-cn-data

    #     trading_date: str
    #         trading days to be updated, by default ``datetime.datetime.now().strftime("%Y-%m-%d")``
    #     end_date: str
    #         end datetime, default ``pd.Timestamp(trading_date + pd.Timedelta(days=1))``; open interval(excluding end)
    #     check_data_length: int
    #         check data length, if not None and greater than 0, each symbol will be considered complete if its data length is greater than or equal to this value, otherwise it will be fetched again, the maximum number of fetches being (max_collector_count). By default None.
    #     delay: float
    #         time.sleep(delay), default 1
    #     Notes
    #     -----
    #         If the data in qlib_data_dir is incomplete, np.nan will be populated to trading_date for the previous trading day

    #     Examples
    #     -------
    #         $ python collector.py update_data_to_bin --qlib_data_1d_dir <user data dir> --trading_date <start date> --end_date <end date>
    #         # get 1m data
    #     """

    #     if self.interval.lower() != "1d":
    #         logger.warning(f"currently supports 1d data updates: --interval 1d")

    #     # start/end date
    #     if trading_date is None:
    #         trading_date = datetime.datetime.now().strftime("%Y-%m-%d")
    #         logger.warning(f"trading_date is None, use the current date: {trading_date}")

    #     if end_date is None:
    #         end_date = (pd.Timestamp(trading_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    #     # download qlib 1d data
    #     qlib_data_1d_dir = str(Path(qlib_data_1d_dir).expanduser().resolve())
    #     if not exists_qlib_data(qlib_data_1d_dir):
    #         GetData().qlib_data(target_dir=qlib_data_1d_dir, interval=self.interval, region=self.region)

    #     # download data from yahoo
    #     # NOTE: when downloading data from YahooFinance, max_workers is recommended to be 1
    #     self.download_data(delay=delay, start=trading_date, end=end_date, check_data_length=check_data_length)
    #     # NOTE: a larger max_workers setting here would be faster
    #     self.max_workers = (
    #         max(multiprocessing.cpu_count() - 2, 1)
    #         if self.max_workers is None or self.max_workers <= 1
    #         else self.max_workers
    #     )
    #     # normalize data
    #     self.normalize_data_1d_extend(qlib_data_1d_dir)

    #     # dump bin
    #     _dump = DumpDataUpdate(
    #         csv_path=self.normalize_dir,
    #         qlib_dir=qlib_data_1d_dir,
    #         exclude_fields="symbol,date",
    #         max_workers=self.max_workers,
    #     )
    #     _dump.dump()

    #     # parse index
    #     _region = self.region.lower()
    #     if _region not in ["cn", "us"]:
    #         logger.warning(f"Unsupported region: region={_region}, component downloads will be ignored")
    #         return
    #     index_list = ["CSI100", "CSI300"] if _region == "cn" else ["SP500", "NASDAQ100", "DJIA", "SP400"]
    #     get_instruments = getattr(
    #         importlib.import_module(f"data_collector.{_region}_index.collector"), "get_instruments"
    #     )
    #     for _index in index_list:
    #         get_instruments(str(qlib_data_1d_dir), _index)


if __name__ == "__main__":
    fire.Fire(Run)
