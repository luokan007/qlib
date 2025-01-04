import os
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import baostock as bs
from contextlib import redirect_stdout

def _login_baostock() -> None:
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull):
            lg = bs.login()
            if lg.error_code != '0':
                print(f"Login failed: {lg.error_msg}")

def _logout_baostock() -> None:
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull):
            bs.logout()

def _fetch_data(query_func, code: str, year: int, quarter: int) :
    rs = query_func(code=code, year=year, quarter=quarter)
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    if not data_list:
        return pd.DataFrame(columns=rs.fields)
    return pd.DataFrame(data_list, columns=rs.fields)

def _convert_quarterly_to_daily(quarterly_df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if quarterly_df.empty:
        return pd.DataFrame(index=pd.date_range(start=start_date, end=end_date))
    dt_start = pd.to_datetime(start_date)
    dt_end = pd.to_datetime(end_date)
    orig_start_date = quarterly_df.index.min()
    orig_end_date = quarterly_df.index.max()
    
    if dt_start > orig_start_date:
        dt_start = orig_start_date
    
    if dt_end < orig_end_date:
        dt_end = orig_end_date
    # Create a complete date range from start_date to end_date
    full_date_range = pd.date_range(start=dt_start, end=dt_end)

    # Resample to daily frequency and forward fill

    daily_df = quarterly_df.resample('D').ffill()

    # Reindex to ensure the same date range as the original data
    daily_df = daily_df.reindex(full_date_range)

    return daily_df

def _filter_quarterly_df(df: pd.DataFrame, target_columns) -> pd.DataFrame:
    col = ['pubDate'] + target_columns
    df = df[col]
    df = df.rename(columns={'pubDate': 'date'})
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df.fillna(method="ffill").fillna(0, inplace=True)
    return df

def _fetch_and_merge_quarterly_data(code: str, raw_df: pd.DataFrame, intermediate_dir: Path) -> pd.DataFrame:
    _login_baostock()
    
    start_date = raw_df.index.min().strftime('%Y-%m-%d')
    end_date = raw_df.index.max().strftime('%Y-%m-%d')
    
    all_quarterly_dfs = []
    growth_columns = ['YOYEquity', 'YOYAsset','YOYNI','YOYEPSBasic','YOYPNI']
    operation_columns = ['NRTurnRatio','NRTurnDays','INVTurnRatio','INVTurnDays','CATurnRatio','AssetTurnRatio']
    balance_columns = ['YOYLiability','liabilityToAsset','assetToEquity']
    cashflow_columns = ['CAToAsset','NCAToAsset','tangibleAssetToAsset','CFOToOR','CFOToNP','CFOToGr']
    dupont_columns = ['dupontROE','RdupontAssetStoEquityOA','dupontAssetTurn','dupontPnitoni']

    # step 1: Fetch growth data
    # Fetch growth data
    growth_dfs = []
    for year in range(int(start_date[:4]) - 1, int(end_date[:4]) + 1):
        for quarter in range(1, 5):
            df_growth = _fetch_data(bs.query_growth_data, code, year, quarter)
            growth_dfs.append(df_growth)
    df_growth = pd.concat(growth_dfs, ignore_index=True)
    df_growth = _filter_quarterly_df(df_growth, growth_columns)
    all_quarterly_dfs.append(df_growth)
    
    # step 2: Fetch operation data
    operation_list = []
    for year in range(int(start_date[:4]) - 1, int(end_date[:4]) + 1):
        for quarter in range(1, 5):
            _operation = _fetch_data(bs.query_operation_data, code, year, quarter)
            operation_list.append(_operation)
    df_operation = pd.DataFrame(operation_list, columns=["pubDate"] + operation_columns)
    df_operation = _filter_quarterly_df(df_operation, operation_columns)
    all_quarterly_dfs.append(df_operation)
    
    # step 3： Fetch balance data
    balance_list = []
    for year in range(int(start_date[:4]) - 1, int(end_date[:4]) + 1):
        for quarter in range(1, 5):
            _balance = _fetch_data(bs.query_balance_data, code, year, quarter)
            balance_list.append(_balance)
    
    df_balance = pd.DataFrame(balance_list, columns=["pubDate"] + balance_columns)
    df_balance = _filter_quarterly_df(df_balance, balance_columns)
    all_quarterly_dfs.append(df_balance)
    
    # step 4: Fetch cash flow data
    cash_flow_list = []
    for year in range(int(start_date[:4]) - 1, int(end_date[:4]) + 1):
        for quarter in range(1, 5):
            _cash_flow = _fetch_data(bs.query_cash_flow_data, code, year, quarter)
            cash_flow_list.append(_cash_flow)
    df_cash_flow = pd.DataFrame(cash_flow_list, columns=["pubDate"] + cashflow_columns)
    df_cash_flow = _filter_quarterly_df(df_cash_flow, cashflow_columns)
    all_quarterly_dfs.append(df_cash_flow)
    
    # step 5: Fetch dupont data
    dupont_list = []
    for year in range(int(start_date[:4]) - 1, int(end_date[:4]) + 1):
        for quarter in range(1, 5):
            _dupont = _fetch_data(bs.query_dupont_data, code, year, quarter)
            dupont_list.append(_dupont)
    df_dupont = pd.DataFrame(dupont_list, columns=["pubDate"] + dupont_columns)
    df_dupont = _filter_quarterly_df(df_dupont, dupont_columns)
    all_quarterly_dfs.append(df_dupont)

    _logout_baostock()

    # Concatenate all quarterly data into a single DataFrame
    all_quarterly_df = pd.concat(all_quarterly_dfs, axis=0, sort=True)
    
    # Save the concatenated quarterly data to a CSV file
    output_file_path = intermediate_dir / f"{code}_quarterly.csv"
    all_quarterly_df.to_csv(output_file_path, encoding="gbk", index=False)
    
    # Remove duplicate columns before merging
    dfs = [_convert_quarterly_to_daily(df, start_date, end_date) for df in all_quarterly_dfs]

    # Remove duplicate columns before merging
    dfs = [_convert_quarterly_to_daily(df, start_date, end_date) for df in all_quarterly_dfs]
    unique_dfs = []
    seen_columns = set()
    
    for df in dfs:
        new_columns = [col for col in df.columns if col not in seen_columns]
        unique_dfs.append(df[new_columns])
        seen_columns.update(new_columns)
    
    # Merge all quarterly data into one DataFrame
    merged_quarterly_df = pd.concat(unique_dfs, axis=1)
    
    # Merge with the original raw data
    final_df = pd.merge(raw_df, merged_quarterly_df, how='left', left_index=True, right_index=True)

    return final_df

def _process_stock_file(file_path: str, intermediate_dir: Path) -> None:
    file_name = Path(file_path).stem
    code = f"{file_name[:2]}.{file_name[2:]}"
    
    raw_df = pd.read_csv(file_path, index_col=0, parse_dates=['date'])
    
    processed_df = _fetch_and_merge_quarterly_data(code, raw_df, intermediate_dir)
    
    output_file_path = f"{Path(file_path).parent.parent}/processed/{file_name}.csv"
    os.makedirs(Path(output_file_path).parent, exist_ok=True)
    processed_df.to_csv(output_file_path, encoding="gbk", index=True)

def main(raw_folder: str, max_workers: int = 1) -> None:
    raw_folder_path = Path(raw_folder)
    intermediate_folder_path = raw_folder_path.parent / "intermediate"
    intermediate_folder_path.mkdir(parents=True, exist_ok=True)
    
    files = [str(f) for f in raw_folder_path.glob("*.csv")]
    
    with tqdm(total=len(files)) as pbar:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_stock_file, file, intermediate_folder_path) for file in files]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing {file}: {e}")
                pbar.update(n=1)

if __name__ == "__main__":
    raw_folder = "/home/godlike/project/GoldSparrow/Day_Data/Day_data/Raw" # 替换为实际的raw文件夹路径
    main(raw_folder)