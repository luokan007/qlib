# 增补后续新数据
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import os
import shutil
import datetime
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout

import baostock as bs
from baostock.data.resultset import ResultData
import akshare as ak

# from qlib_dump_bin import DumpDataAll
from mydump_bin import DumpDataAll, DumpDataUpdate


def _read_all_text(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def _write_all_text(path: str, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)


class DataManager:
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
        "peTTM",
        "psTTM",
        "pcfNcfTTM",
        "pbMRQ",
        "isST",
    ]
    _price_fields: List[str] = ["open", "high", "low", "close", "preclose"]

    def __init__(
        self,
        csv_path: str = None,
        qlib_data_path: str = None,
        adjustflag: str = "1",  # "3": 不复权；"1"：后复权； "2"：前复权。
        start_date: str = None,  # 下载数据开始日期，格式如"2015-01-01" ，None从上市日开始
        end_date: str = None,  # 下载数据的结束日期。None则到最近日
        overwrite: bool = True,
        max_workers: int = 5,
    ):
        self._csv_path = os.path.expanduser(csv_path)
        os.makedirs(self._csv_path, exist_ok=True)

        self._qlib_data_path = os.path.expanduser(qlib_data_path)
        self._adjustflag = adjustflag

        self._start_date = start_date
        self._end_date = end_date
        
        self._latest_trading_date = str(datetime.date.today()) # 初始最近交易日

        self._overwrite = overwrite
        self._max_workers = max_workers
       
        self._all_a_shares = [] # 本次更新股票列表全集       
        self._new_a_shares = [] # 上次更新数据后，新上市的股票列表

        df_old_shares = pd.read_csv(f"{self._qlib_data_path}/instruments/all.txt",sep="\t",header=None)

        # 上次更新行情数据时存在的股票列表
        self._old_a_shares = [f"{s[:2].lower()}.{s[-6:]}" for s in df_old_shares[0]]
        # print(self._old_a_shares)
        # raise Exception('qq')

    @classmethod
    def _login_baostock(cls) -> None:
        with open(os.devnull, "w") as devnull:
            with redirect_stdout(devnull):
                bs.login()

    @property
    def _a_shares_list_path(self) -> str:
        return f"{self._qlib_data_path}/a_shares_list.txt"

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
        # 1 先从已有股票代码列表文件获取股票代码列表放进self._all_a_shares，这个应该含历史退市股，这一步是防止防止幸存者偏差的关键。这个文件可以只含历史退市股，从问财可以查到全部历史退市股
        # 文件中代码格式 sh600000，或sh.600000
        # 此步省略影响不大
        self._load_all_a_shares_base()

        # 再从在线baostock api获取最近交易日的活跃股票代码列表，代码格式类似
        for i in range(1000):
            df = bs.query_all_stock(
                day=str(datetime.date.today() - datetime.timedelta(days=i))
            ).get_data()
            if len(df)==0:
                continue
            else:
                print("get data until",str(datetime.date.today() - datetime.timedelta(days=i)), "active stock num", len(df))
                self._latest_trading_date = str(datetime.date.today() - datetime.timedelta(days=i)) # 最近有效交易日
                break
            
        


        # 2 活跃股票代码集合
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
        # 3 获取退市股代码，幸存者无偏
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
        )  # 历史退市股合并指数后，合并活跃股票，形成最终股票列表全集
        _write_all_text(
            self._a_shares_list_path,
            "\n".join(str(s[:2] + s[-6:]) for s in self._all_a_shares),
        )  # 覆盖写到代码列表文件,去掉.号，文件里代码格式sh600000

        self._new_a_shares = list(set(self._all_a_shares) - set(self._old_a_shares))
        print("self._new_a_shares",self._new_a_shares)



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
        return self._adjust_factors.xs(code, level="code").astype(float)  # type: ignore

    def _download_stock_data_job(self, code: str, data: pd.Series) -> None:
        fields_str = ",".join(self._fields)
        numeric_fields = self._fields.copy()
        numeric_fields.pop(0)
        # print("code",code)
        
        self._login_baostock()
        # 获取复权数据
        # print("qqq",self._start_date ,"ipo", data["ipoDate"])
        query = bs.query_history_k_data_plus(
            code,
            fields_str,
            # 新股从ipo开始下载，老股从设置的开始日期下载
            start_date=data["ipoDate"] if code in self._new_a_shares else self._start_date, 
         
            end_date=self._end_date,
            adjustflag=self._adjustflag,
        )
        #  adjustflag：复权类型，
        # 3: 默认不复权；
        # 1：后复权；
        # 2：前复权。已支持分钟线、日线、周线、月线前后复权。 BaoStock提供的是涨跌幅复权算法复权因子，具体介绍见：复权因子简介或者BaoStock复权因子简介。
        adj = self._adjust_factors_for(code)  # 获取复权因子
        df = query.get_data()
        if df.empty:
            print(code,"is empty")
            return
      
        df = df[df.tradestatus == "1"]  # 筛出未停牌的记录

        df = df.join(adj, on="date", how="left")
        df[self._adjust_columns] = (
            df[self._adjust_columns].fillna(method="ffill").fillna(1.0)
        )  # 复权因子列，前后因子都有
        df[numeric_fields] = df[numeric_fields].replace("", "0.").astype(float)
        df = df.set_index("date")

        df["factor"] = (
            df["backAdjustFactor"]
            if self._adjustflag == "1"
            else df["foreAdjustFactor"]
        )
        df["volume"] /= df["factor"]
        df["vwap"] = df["amount"] / df["volume"]

        df.to_csv(
            f"{self._csv_path}/{code[:2].lower()}{code[-6:]}.csv"
        )  # 保存的csv文件代码不带.,类似sh600000.csv

    def _download_stock_data(self) -> None:
        print("Download stock data")
        os.makedirs(f"{self._csv_path}", exist_ok=True)

        # 获取目录中已下载csv文件的股票代码列表 qtb add
        csv_file_list = os.listdir(f"{self._csv_path}")
        self._code_downloaded = []
        if not self._overwrite:  # 如果不覆盖已下载的数据，则跳过已下载的
            code_downloaded = [f[-13:-4] for f in csv_file_list]  # 记录已下载的股票代码列表
            self._code_downloaded = [c[:2] + "." + c[-6:] for c in code_downloaded]

        # i = 0
        # for code, data in self._basic_info.iterrows():
        #     if code not in self._code_downloaded:
        #         i+=1
        #         print(i,code)
        #         self._download_stock_data_job(code, data)


        # 多线程下载
        # code=code,	data= code_name	ipoDate	outDate	type	status
        self._parallel_foreach(
            self._download_stock_data_job,
            [
                dict(code=code, data=data)
                for code, data in self._basic_info.iterrows()
                if code not in self._code_downloaded
            ],
        )



    @classmethod
    def _result_to_data_frame(cls, res: ResultData) -> pd.DataFrame:
        lst = []
        while res.error_code == "0" and res.next():
            lst.append(res.get_row_data())
        return pd.DataFrame(lst, columns=res.fields)

    def _dump_qlib_data(self) -> None:
        print("dump qlib data")
        DumpDataUpdate(
            csv_path=self._csv_path,
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
        # 更新成分股池最新日期
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

        self._download_stock_data()

        self._dump_qlib_data()



if __name__ == "__main__":
    # 下载最新的全市场股票行情
    today = str(datetime.date.today())
    print(f"today: {today}")
    qlib_data_path=r"G:/qlibrolling/qlib_data/cn_data_rolling"

    # 获取上次更新日期
    cal_df = pd.read_csv(qlib_data_path+"/calendars/day.txt",header=None) # 读取日历
    last_date = cal_df[0].max() # 上次日历最后日期，即现有行情数据最后日期。字符串


    start_date = str(pd.to_datetime(last_date)+pd.to_timedelta("1 days"))[:10] # 上次数据结束日期+1天
    print("down data from",start_date)
    
    dm = DataManager(
        csv_path= qlib_data_path + "/csv_append",
        qlib_data_path=qlib_data_path,
        start_date=start_date,  # 下载数据开始日期，格式如"2015-01-01" ，None从上市日开始
        end_date=None,  # 下载数据的结束日期。None则到最近日
        #  adjustflag：复权方式，字符串."3": 不复权；"1"：后复权；"2"：前复权。
        #  BaoStock提供的是涨跌幅复权算法复权因子，具体介绍见：BaoStock复权因子简介。
        adjustflag="1",
        overwrite=True,  # 是否覆盖已存在的股票行情csv文件
        max_workers=5,
    )

    useCache = False  # 使用缓存股票基本信息，和调整因子。入股股票代码中的代码在缓存基本信息中不存在，则不会下载其行情数据
    dm.fetch_and_save_data(
        use_cached_basic_info=useCache, use_cached_adjust_factor=useCache
    )
