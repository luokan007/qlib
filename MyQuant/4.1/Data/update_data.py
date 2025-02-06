# title: get_baostock_data
# updated: 2024.12.17


# 目标：
#   1. 从baostock获取全量数据
#   - 支持自定义时间区间，默认为最近一年
#   - 扩展当前的feature范围

import os
import shutil
import datetime
from pathlib import Path
from contextlib import redirect_stdout
from typing import Tuple, List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
import pandas as pd
import baostock as bs
from baostock.data.resultset import ResultData
import akshare as ak

# from qlib_dump_bin import DumpDataAll
from ta_lib_feature import TALibFeatureExt
from mydump_bin import DumpDataAll
import warnings
warnings.filterwarnings("ignore")

def _read_all_text(path: str) -> str:
    with open(path, "r") as f:
        return f.read()

def _write_all_text(path: str, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)

class EnhancedDataManager:
    _all_a_shares: List[str]
    _basic_info: pd.DataFrame
    _adjust_factors: pd.DataFrame

    _adjust_columns: List[str] = [
        "foreAdjustFactor",
        "backAdjustFactor",
        "adjustFactor",
    ]
    _fields: List[str] = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "preclose",
        "volume",
        "amount",
        "turn",
        "tradestatus",
        "pctChg",
        "isST",
    ]
    _price_fields: List[str] = ["open", "high", "low", "close", "preclose"]

    def __init__(
        self,
        csv_path: str = None,
        feature_path: str = None,
        qlib_data_path: str = None,
        basic_info_path: str = None,
        feature_meta_file: str = None,
        stock_pool_file: str = None,
        adjustflag: str = "1",  # "3": 不复权；"1"：后复权； "2"：前复权。
        start_date: str = None,  # 下载数据开始日期，格式如"2015-01-01" ，None从上市日开始
        end_date: str = None,  # 下载数据的结束日期。None则到最近日
        overwrite: bool = True,
        max_workers: int = 5,
    ):
        self._csv_path = os.path.expanduser(csv_path)
        
        self._feature_path = os.path.expanduser(feature_path)
        #删除原有的特征文件夹
        if os.path.exists(self._feature_path) and os.path.isdir(self._feature_path):
            shutil.rmtree(self._feature_path)
        os.makedirs(self._feature_path, exist_ok=True)

        self._qlib_data_path = os.path.expanduser(qlib_data_path)
        self._basic_info_path = os.path.expanduser(basic_info_path)
        self._feature_meta_file = feature_meta_file
        self._stock_pool_file = stock_pool_file
        
        self._adjustflag = adjustflag

        self._start_date = start_date
        self._end_date = end_date
        self._latest_trading_date = str(datetime.date.today()) # 初始最近交易日

        self._overwrite = overwrite
        self._max_workers = max_workers
        self._all_a_shares = []

    @classmethod
    def _login_baostock(cls) -> None:
        with open(os.devnull, "w") as devnull:
            with redirect_stdout(devnull):
                bs.login()

    @property
    def _a_shares_list_path(self) -> str:
        return f"{self._qlib_data_path}/a_shares_list.txt"
    
    def update_time_range(self, input_dir: str):
        MODE = "create"
        if not os.path.exists(input_dir):
            today = datetime.date.today().strftime("%Y-%m-%d")
            return (MODE, "2008-01-01", today)
        else:
            all_start_dates = []
            all_end_dates = []
            for file_path in Path(input_dir).glob('*.csv'):
                df = pd.read_csv(file_path, usecols=["date"])
                if not df.empty: 
                    all_start_dates.append(df["date"].min())
                    all_end_dates.append(df["date"].max())
            
            if all_start_dates and all_end_dates:
                min_existing = min(all_start_dates)
                max_existing = max(all_end_dates)
                
                assert min_existing <= max_existing, "Invalid time range"
                assert max_existing <= datetime.date.today().strftime("%Y-%m-%d"), "Existing data end date is in the future"
                
                today = datetime.date.today().strftime("%Y-%m-%d")
                if max_existing < today:
                    max_existing_plus_one = (pd.to_datetime(max_existing) + pd.DateOffset(days=1)).strftime("%Y-%m-%d")
                    MODE = "update"
                    return (MODE, max_existing_plus_one, today)
                elif max_existing == today:
                    MODE = "skip"
                    return (MODE, None, None)
                else:
                    raise ValueError("Invalid time range")
            else:
                raise ValueError("No data found")   

    def _load_all_a_shares_base(self) -> None:
        # 从已有股票代码列表文件获取股票代码列表，含历史退市股即可，防止幸存者偏差。最终会与在线拉下来的活跃股求并集
        if os.path.exists(self._a_shares_list_path):
            lines = _read_all_text(self._a_shares_list_path).split("\n")
            all_a_shares = [line for line in lines if line != ""]

            if len(all_a_shares) == 0:
                return
            if "." not in all_a_shares[0]:  # 股票代码不含.的要转成类似sh.600000,以便baostock api接受
                self._all_a_shares = [
                    f"{stk_id[:2].lower()}.{stk_id[-6:]}" for stk_id in all_a_shares
                ]
            else:  # 已经带.的代码转小写即可
                self._all_a_shares = [f"{stk_id.lower()}" for stk_id in all_a_shares]

    def _load_all_a_shares(self):
        print("Loading A-Shares stock list")
        # 获取活跃股票代码列表（不含退市的）以及指数代码列表,并合并初始股票列表
        self._login_baostock()
        # 先从已有股票代码列表文件获取股票代码列表放进self._all_a_shares，这个应该含历史退市股，这一步是防止防止幸存者偏差的关键。这个文件可以只含历史退市股，从问财可以查到全部历史退市股
        # 文件中代码格式 sh600000，或sh.600000
        self._load_all_a_shares_base()

        # 再从在线baostock api获取最近的活跃股票代码列表，代码格式类似
        for i in range(1000):
            df = bs.query_all_stock(
                day=str(datetime.date.today() - datetime.timedelta(days=i))
            ).get_data()
            if len(df)==0:
                continue
            else:
                print("get data until",str(datetime.date.today() - datetime.timedelta(days=i)), "active stock num", len(df))
                self._latest_trading_date = str(datetime.date.today() - datetime.timedelta(days=i)) # 最近交易日
                break
       
        # 活跃股票代码集合
        stocks = {
            code
            for code in df["code"]
            if code.startswith("sh") or code.startswith("sz")
        }

        self._all_a_shares += [
            "sh.000903",
            "sh.000300",
            "sh.000905",
            "sh.000852",
        ]  # 指数集合

        ############
        # 获取退市股代码，幸存者无偏
        ###############
        try:
            sh_delist = {
                "sh." + s for s in ak.stock_info_sh_delist()["公司代码"] if s[0] == "6"
            }  # 上海退市A股。不考虑B股
        except:
            sh_delist = set()
        try:
            sz_delist = {
                "sz." + s
                for s in ak.stock_info_sz_delist("终止上市公司")["证券代码"]
                if s[0] in ["0", "3"]
            }  # 生成退市A股和创业板股3打头
        except:
            sz_delist = set()

        self._all_a_shares = list(
            set(self._all_a_shares).union(stocks, sh_delist, sz_delist)
        )  # 历史退市股合并指数后，合并活跃股票，形成最终股票列表
        _write_all_text(
            self._a_shares_list_path,
            "\n".join(str(s[:2] + s[-6:]) for s in self._all_a_shares),
        )  # 写到代码列表文件,去掉.号，文件里代码格式sh600000

    def _parallel_foreach(
        self, callable, input: List[dict], max_workers: Optional[int] = None
    ) -> list:
        if max_workers is None:
            max_workers = self._max_workers
        with tqdm(total=len(input)) as pbar:
            results = []
            with ProcessPoolExecutor(max_workers) as executor:
                futures = [executor.submit(callable, **elem) for elem in input]
                for f in as_completed(futures):
                    results.append(f.result())
                    pbar.update(n=1)
            return results

    def _fetch_basic_info_job(self, code: str) -> pd.DataFrame:
        self._login_baostock()
        return bs.query_stock_basic(code).get_data()

    def _fetch_basic_info(self) -> pd.DataFrame:
        print("Fetching basic info")

        dfs = self._parallel_foreach(
            self._fetch_basic_info_job, [dict(code=code) for code in self._all_a_shares]
        )
        df = pd.concat(dfs)
        df = df.sort_values(by="code").drop_duplicates(subset="code").set_index("code")
        df.to_csv(f"{self._qlib_data_path}/basic_info.csv")
        return df

    def _fetch_adjust_factors_job(self, code: str, start: str) -> pd.DataFrame:
        self._login_baostock()
        return bs.query_adjust_factor(code, start).get_data()

    def _fetch_adjust_factors(self) -> pd.DataFrame:
        def one_year_before_ipo(ipo: str) -> str:
            earliest_time = pd.Timestamp("1990-12-19")
            ts = pd.Timestamp(ipo) - pd.DateOffset(years=1)
            ts = earliest_time if earliest_time > ts else ts
            return ts.strftime("%Y-%m-%d")

        print("Fetch adjust factors")
        dfs: List[pd.DataFrame] = self._parallel_foreach(
            self._fetch_adjust_factors_job,
            [
                dict(code=code, start=one_year_before_ipo(data["ipoDate"]))
                for code, data in self._basic_info.iterrows()
            ],
        )
        df = pd.concat([df for df in dfs if not df.empty])
        df = df.set_index(["code", "dividOperateDate"])
        df.to_csv(f"{self._qlib_data_path}/adjust_factors.csv")
        return df

    def _adjust_factors_for(self, code: str) -> pd.DataFrame:
        adj_factor_idx: pd.Index = self._adjust_factors.index.levels[0]  # type: ignore
        if code not in adj_factor_idx:
            start: str = self._basic_info.loc[code, "ipoDate"]  # type: ignore
            return pd.DataFrame(
                [[1.0, 1.0, 1.0]], index=pd.Index([start]), columns=self._adjust_columns
            )
        ret_df = self._adjust_factors.xs(code, level="code").astype(float)  # type: ignore
        #print(ret_df.tail())
        if ret_df.index.max() < self._start_date:##当调整因子的最大日期小于开始日期时，需要补充调整因子
            new_row = pd.DataFrame([[float('nan')] * len(self._adjust_columns)], 
                       index=pd.Index([self._start_date]), 
                       columns=self._adjust_columns)
            ret_df = pd.concat([ret_df, new_row])
            ret_df = ret_df.fillna(method='ffill')
            #print(ret_df.tail())
        return ret_df
        #return self._adjust_factors.xs(code, level="code").astype(float)  # type: ignore

    def _check_csv_data(self, code: str) -> Tuple[bool, bool, pd.DataFrame]:
        """
        检查CSV文件是否存在且包含所需数据
        Returns:
            Tuple[bool, bool, pd.DataFrame]: 
            - 第一个bool表示是否包含日频数据
            - 第二个bool表示是否包含估值数据
            - DataFrame为读取的数据
        """
        file_path = Path(self._csv_path) / f"{code}.csv"
        if not file_path.exists():
            return False, False, None
            
        try:
            df = pd.read_csv(file_path, index_col=0)
            df.index = pd.to_datetime(df.index)
            
            # 检查日期范围
            if not (df.index.min().strftime('%Y-%m-%d') <= self._start_date and
                    df.index.max().strftime('%Y-%m-%d') >= self._end_date):
                return False, False, None
                
            # 检查日频数据列
            daily_columns = ["date","open","high","low","close",
                             "preclose","volume","amount","turn",
                             "tradestatus","pctChg","isST"]
            has_daily = all(col in df.columns for col in daily_columns)
            
            # 检查估值数据列
            valuation_columns = ['peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM']
            has_valuation = all(col in df.columns for col in valuation_columns)
            
            return has_daily, has_valuation, df
        except Exception as e:
            return False, False, None

    def _download_daily_data(self, code: str) -> pd.DataFrame:
        """下载日频数据"""
        fields = "date,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,isST"
        rs = bs.query_history_k_data_plus(
            code, fields, start_date=self._start_date, end_date=self._end_date, adjustflag=self._adjustflag,
        )
        return self._result_to_data_frame(rs)

    def _download_valuation_data(self, code: str) -> pd.DataFrame:
        """下载估值数据"""
        fields = "date,peTTM,pbMRQ,psTTM,pcfNcfTTM"
        rs = bs.query_history_k_data_plus(
            code, fields, start_date=self._start_date, end_date=self._end_date, adjustflag=self._adjustflag,
        )
        return self._result_to_data_frame(rs)

    def _download_seasonal_data(self, code: str) -> pd.DataFrame:
        """下载季度财务数据"""
        start_year = int(self._start_date.split('-')[0])
        end_year = int(self._end_date.split('-')[0]) if self._end_date else datetime.datetime.now().year

        # Start from previous year to ensure data for early dates
        profit_list = []
        for year in range(start_year - 1, end_year + 1):
            for quarter in range(1, 5):
                rs = bs.query_profit_data(code=code, year=year, quarter=quarter)
            
                while (rs.error_code == '0') and rs.next():
                    profit_list.append(rs.get_row_data())

        if not profit_list:
            return pd.DataFrame()
            
        return pd.DataFrame(profit_list, columns=rs.fields)

    def _download_performance_data(self, code: str) -> pd.DataFrame:
        """下载业绩快报数据"""
        start_date = (pd.to_datetime(self._start_date) - pd.DateOffset(years=1)).strftime('%Y-%m-%d')

        rs = bs.query_performance_express_report(
            code=code,
            start_date=start_date,
            end_date=self._end_date
        )

        result_list = []
        while (rs.error_code == '0') and rs.next():
            result_list.append(rs.get_row_data())

        if not result_list:
            return pd.DataFrame()

        df = pd.DataFrame(result_list, columns=rs.fields)

        numeric_columns = [
            'performanceExpressTotalAsset',
            'performanceExpressNetAsset',
            'performanceExpressEPSChgPct',
            'performanceExpressROEWa',
            'performanceExpressEPSDiluted',
            'performanceExpressGRYOY',
            'performanceExpressOPYOY'
        ]

        # 重命名日期列并设置为索引
        df = df[['performanceExpPubDate'] + numeric_columns]
        df = df.rename(columns={'performanceExpPubDate': 'date'})
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        # 转换数值类型
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def _convert_performance_to_daily(self, performance_df: pd.DataFrame) -> pd.DataFrame:
        """将业绩快报数据转换为日频数据"""
        if performance_df.empty:
            return performance_df
            
        # 重采样为日频数据并前向填充
        daily_df = performance_df.resample('D').ffill()
        return daily_df


    def _convert_seasonal_to_daily(self, seasonal_df: pd.DataFrame) -> pd.DataFrame:
        """将季度数据转换为日频数据"""
        seasonal_df['date'] = pd.to_datetime(seasonal_df['pubDate'])
        seasonal_df.set_index('date', inplace=True)

        # 选择需要的财务指标列
        financial_columns = ['epsTTM', 'totalShare', 'liqaShare']
        seasonal_df = seasonal_df[financial_columns]
        seasonal_df.index = pd.to_datetime(seasonal_df.index)
        try:
            seasonal_df = seasonal_df[~seasonal_df.index.duplicated()].resample('D').ffill()  # Remove duplicates before resampling
        except Exception as e:
            print(f"Failed to resample seasonal data: {e}")
            return pd.DataFrame()

        return seasonal_df

    def _process_stock_data(self, code: str, data: pd.Series, target_folder) -> None:
        """Process stock data for a given code: download daily data, valuation data, and merge them."""
        self._login_baostock()

        has_daily, has_valuation, existing_df = self._check_csv_data(code)

        # 1. 处理日频数据
        if not has_daily:
            df_daily = self._download_daily_data(code)
            
            if df_daily is None or df_daily.empty:
                return
        else:
            df_daily = existing_df

        df_daily = df_daily.set_index("date")

        # 2. 处理估值数据
        if not has_valuation:
            df_valuation = self._download_valuation_data(code)
            if df_valuation.empty:
                df_valuation = pd.DataFrame(columns=["date", "code", "peTTM", "psTTM", "pcfNcfTTM", "pbMRQ"])
                
            df_valuation = df_valuation.set_index("date")
            merged_df = pd.merge(df_daily, df_valuation, how='left', left_index=True, right_index=True)
            merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]  # Remove duplicate columns
        else:
            merged_df = df_daily
        
        # Ensure all valuation columns are present and fill NaNs with 0
        for col in ['peTTM', 'psTTM', 'pcfNcfTTM', 'pbMRQ']:
            if col not in merged_df.columns:
                merged_df[col] = 0
            else:
                merged_df[col].fillna(0, inplace=True)

        daily_columns =["open","high","low","close",
                        "preclose","volume","amount",
                        "turn","tradestatus","pctChg","isST"]
        valuation_columns = ['peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM']

        merged_df[daily_columns] = merged_df[daily_columns].replace("", "0.").astype(float)
        merged_df[valuation_columns] = merged_df[valuation_columns].replace("", "0.").astype(float)

        adj = self._adjust_factors_for(code)  # 获取复权因子
        adj.index.name = "date"
        min_date, max_date = merged_df.index.min(), merged_df.index.max()
        if min_date not in adj.index:
            adj.loc[min_date] = [float('nan')] * len(self._adjust_columns)
        if max_date not in adj.index:
            adj.loc[max_date] = [float('nan')] * len(self._adjust_columns)
        adj = adj.sort_index().fillna(method='ffill')

        merged_df = merged_df[merged_df.tradestatus == 1]  # 筛出未停牌的记录
        merged_df = merged_df.join(adj, on="date", how="left")
        merged_df[self._adjust_columns] = (
            merged_df[self._adjust_columns].fillna(method="ffill").fillna(1.0))  # 复权因子列，前后因子都有


        merged_df["factor"] = (
            merged_df["backAdjustFactor"]
            if self._adjustflag == "1"
            else merged_df["foreAdjustFactor"]
        )
        merged_df["volume"] /= merged_df["factor"]
        merged_df["vwap"] = merged_df["amount"] / merged_df["volume"]

        # 4. 下载并合并季度数据
        seasonal_columns = ['epsTTM', 'totalShare', 'liqaShare']
        seasonal_df = self._download_seasonal_data(code)
        if seasonal_df is not None and not seasonal_df.empty:
            # 将季度数据转换为日频
            seasonal_df = self._convert_seasonal_to_daily(seasonal_df)
            try:
                merged_df.index = pd.to_datetime(merged_df.index)
                seasonal_df.index = pd.to_datetime(seasonal_df.index)
                merge_min_date, merge_max_date = merged_df.index.min(), merged_df.index.max()
                seasonal_min_date, seasonal_max_date = seasonal_df.index.min(), seasonal_df.index.max()
                
                ##细分几种情况
                if merge_min_date < seasonal_min_date:
                    seasonal_df.loc[merge_min_date] = [float('nan')] * len(seasonal_columns)
                    seasonal_df = seasonal_df.sort_index().fillna(method='bfill')
                
                if merge_min_date > seasonal_max_date:
                    seasonal_df.loc[merge_min_date] = [float('nan')] * len(seasonal_columns)
                    seasonal_df = seasonal_df.sort_index().fillna(method='ffill')
                
                merged_df = pd.merge(merged_df, seasonal_df, how='left', left_index=True, right_index=True)
                merged_df[seasonal_columns] = merged_df[seasonal_columns].fillna(method='ffill') 
            except Exception as e:
                print(f"Failed to merge seasonal data for {code}: {e}")
        else:
            seasonal_df = pd.DataFrame(columns=['date', 'epsTTM', 'totalShare', 'liqaShare'])
            seasonal_df = seasonal_df.set_index("date")
            merged_df = pd.merge(merged_df, seasonal_df, how='left', left_index=True, right_index=True)
            merged_df[seasonal_columns] = merged_df[seasonal_columns].fillna(0) #为了避免前向填充的数据被覆盖，这里不再填充

        # Step 6: Save the merged data to CSV
        csv_file_path = f"{target_folder}/{code[:2].lower()}{code[-6:]}.csv"
        if not merged_df.empty:
            try:
                merged_df.to_csv(csv_file_path)
            except Exception as e:
                print(f"Failed to save data for {code}: {e}")

        #bs.logout()
    def _download_stock_data(self, target_folder) -> None:
        print("Download stock data")
        os.makedirs(f"{target_folder}", exist_ok=True)

        # 多线程下载
        #code=code,	data= code_name	ipoDate	outDate	type	status
        self._parallel_foreach(
             self._process_stock_data,
             [
                 dict(code=code, data=data, target_folder=target_folder)
                 for code, data in self._basic_info.iterrows()
             ],
         )

    @classmethod
    def _result_to_data_frame(cls, res: ResultData) -> pd.DataFrame:
        lst = []
        while res.error_code == "0" and res.next():
            lst.append(res.get_row_data())
        return pd.DataFrame(lst, columns=res.fields)

    def _dump_qlib_data(self, csv_folder) -> None:
        print("dump qlib data")
        DumpDataAll(
            csv_path=csv_folder,
            qlib_dir=self._qlib_data_path,
            max_workers=self._max_workers,
            exclude_fields="date,code",
            symbol_field_name="code",
        ).dump()
        shutil.copy(
            f"{self._qlib_data_path}/calendars/day.txt",
            f"{self._qlib_data_path}/calendars/day_future.txt",
        )
        self._fix_constituents()

    def _fix_constituents(self) -> None:
        today = self._latest_trading_date # str(datetime.date.today())
        path = f"{self._qlib_data_path}/instruments"

        for p in Path(path).iterdir():
            if p.stem == "all":
                continue
            df = pd.read_csv(p, sep="\t", header=None)
            df.sort_values(
                [2, 1, 0], ascending=[False, False, True], inplace=True
            )  # type: ignore
            latest_data = df[2].max()
            df[2] = df[2].replace(latest_data, today)
            df.to_csv(p, header=False, index=False, sep="\t")

    def _add_features(self, data_dir, output_dir,basic_info_path):
        """
        Add technical analysis features to stock data using TALib
        
        Parameters:
        -----------
        data_dir : str
            Directory containing the stock data files
        output_dir : str
            Directory to save the processed files with new features
        """
        ta_feature_generator = TALibFeatureExt(basic_info_path=basic_info_path,
                                               time_range=30,
                                                stock_pool_path = self._stock_pool_file)

        if os.path.exists(output_dir) and os.path.isdir(output_dir):
            # 使用 shutil.rmtree 高效地移除整个目录树
            shutil.rmtree(output_dir)

        # # 重新创建目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        ta_feature_generator.process_directory(data_dir, output_dir,self._feature_meta_file)
    
    def fetch_and_save_data(
        self,
        use_cached_basic_info: bool = False,
        use_cached_adjust_factor: bool = False,
    ):
        # 获取活跃股票代码列表合并指数代码列表，放到列表self._all_a_shares
        self._load_all_a_shares()

        # self._all_a_shares = self._all_a_shares[0:5] # qtb add 取几个测试用

        if use_cached_basic_info:
            self._basic_info = pd.read_csv(
                f"{self._qlib_data_path}/basic_info.csv", index_col=0
            )
        else:
            self._basic_info = (
                self._fetch_basic_info()
            )  # 利用_all_a_shares，获取所有股票基本信息，其中也有000300这种指数
        if use_cached_adjust_factor:
            self._adjust_factors = pd.read_csv(
                f"{self._qlib_data_path}/adjust_factors.csv", index_col=[0, 1]
            )
        else:
            self._adjust_factors = self._fetch_adjust_factors()  # 获取所有股票调整因子

        (MODE, start, end ) = self.update_time_range(self._csv_path)    
        if MODE == "skip":
            print("Data up to date, skipping download")
            return
        elif MODE == "create":
            self._start_date = start
            self._end_date = end
            target_folder = self._csv_path
            self._download_stock_data(target_folder=target_folder)
        elif MODE == "update":
            self._start_date = start
            self._end_date = end
            # Create a temporary folder for new data
            temp_folder = os.path.join(self._csv_path, "temp")
            if os.path.exists(temp_folder) and os.path.isdir(temp_folder):
                shutil.rmtree(temp_folder)
            os.makedirs(temp_folder, exist_ok=True)

            # Download new data to the temporary folder
            self._download_stock_data(target_folder=temp_folder)
            self._merge_data(temp_folder=temp_folder, target_folder=self._csv_path)
        else:
            raise ValueError("Invalid mode")
        
        ##生成特征
        self._add_features(self._csv_path, self._feature_path, self._basic_info_path)
        
        ##删除qlib目录下的calendar和features目录
        shutil.rmtree(f"{self._qlib_data_path}/calendars")
        shutil.rmtree(f"{self._qlib_data_path}/features")
        
        ##保存qlib数据
        self._dump_qlib_data(self._feature_path)

    def _merge_data(self, temp_folder: str, target_folder: str) -> None:
        """
        Merge data from the temporary folder to the target folder
        """
        print("Merging data...")
        with tqdm(total=len(list(Path(temp_folder).glob('*.csv')))) as pbar:
            for file_path in Path(temp_folder).glob('*.csv'):
                target_file_path = Path(target_folder) / file_path.name
                temp_df = pd.read_csv(file_path)
                
                if target_file_path.exists():
                    target_df = pd.read_csv(target_file_path)
                    merged_df = pd.concat([target_df, temp_df]).drop_duplicates(subset=['date']).sort_values(by='date')
                    
                    ##seasoanl columns需要前向填充
                    seasonal_columns = ['epsTTM', 'totalShare', 'liqaShare']
                    merged_df[seasonal_columns] = merged_df[seasonal_columns].fillna(method='ffill')
                else:
                    merged_df = temp_df
                
                merged_df.to_csv(target_file_path, index=False)
                pbar.update(1)

        ## Remove the temporary folder
        print("Removing temporary folder")
        shutil.rmtree(temp_folder)
            

if __name__ == "__main__":
    
    # 下载最新的全市场股票行情
    today = str(datetime.date.today())
    print(f"today: {today}")

    # _csv_path=r"/root/autodl-tmp/GoldSparrow/Day_data/Raw"
    # _feature_path=r"/root/autodl-tmp/GoldSparrow/Day_data/Merged_talib"
    # _qlib_data_path=r"/root/autodl-tmp/GoldSparrow/Day_data/qlib_data"
    # _basic_info_path='/root/autodl-tmp/GoldSparrow/Day_data/qlib_data/basic_info.csv'

    ##本地环境
    work_folder = "/home/godlike/project/GoldSparrow/Day_Data/Day_data"

    _csv_path= f"{work_folder}/Raw"
    _feature_path=f"{work_folder}/Merged_talib"
    _qlib_data_path=f"{work_folder}/qlib_data"
    _basic_info_path=f"{work_folder}/qlib_data/basic_info.csv"
    _feature_meta_file = f"{work_folder}/feature_names.json"
    _stock_pool_file = '/home/godlike/project/GoldSparrow/Day_Data/Day_data/qlib_data/instruments/csi300.txt'


    dm = EnhancedDataManager(   

        csv_path=_csv_path,
        feature_path=_feature_path,
        qlib_data_path=_qlib_data_path,
        basic_info_path=_basic_info_path,
        feature_meta_file = _feature_meta_file,
        stock_pool_file = _stock_pool_file,
        #  adjustflag：复权方式，字符串."3": 不复权；"1"：后复权；"2"：前复权。
        #  BaoStock提供的是涨跌幅复权算法复权因子，具体介绍见：BaoStock复权因子简介。
        adjustflag="1",
        overwrite=True,  # 是否覆盖已存在的股票行情csv文件
        max_workers=8,  ##8 core cpu, 云主机为32核
    )

    useCache = True  # 使用缓存股票基本信息，和调整因子。入股股票代码中的代码在缓存基本信息中不存在，则不会下载其行情数据
    dm.fetch_and_save_data(
        use_cached_basic_info=useCache, use_cached_adjust_factor=useCache
    )
