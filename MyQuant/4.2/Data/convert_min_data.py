# filepath: /home/GoldSparrow/qlib/MyQuant/4.0/Data/convert_min_data.py
import json
import pandas as pd
import datetime
from datetime import timedelta
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def _generate_trading_times():
    """Yield从9:30~11:30, 13:01~15:00的分钟级别时间字符串"""
    morning_start = datetime.datetime(2023,1,1,9,30)
    morning_end = datetime.datetime(2023,1,1,11,30)
    afternoon_start = datetime.datetime(2023,1,1,13,1)
    afternoon_end = datetime.datetime(2023,1,1,15,0)
    delta = timedelta(minutes=1)

    t = morning_start
    while t <= morning_end:
        yield t.strftime("%H:%M")
        t += delta

    t = afternoon_start
    while t <= afternoon_end:
        yield t.strftime("%H:%M")
        t += delta

def process_file(args):
    json_file, code_map, output_path = args
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    filename = Path(json_file).stem
    if filename in code_map:
        full_code = code_map[filename]
    else:
        print(f"Cannot find code for {filename}")
    #full_code = code_map.get(filename, filename)

    rows = []
    for date_str, val_list in data.items():
        if len(val_list) < 2:
            continue
        times = _generate_trading_times()
        # partial_code = val_list[0][0]
        # full_code = code_map.get(partial_code, partial_code)

        for idx in range(1, len(val_list)):
            time_str = next(times, None)
            if not time_str:
                break

            open_price = val_list[idx][0]
            vwap = val_list[idx][1]
            volume = val_list[idx][2]

            rows.append({
                'datetime': f"{date_str} {time_str}",
                'open': open_price,
                'vwap': vwap,
                'volume': volume
            })
    df = pd.DataFrame(rows)
    output_file = output_path / f"{full_code}.csv"
    df.to_csv(output_file, index=False)

def convert_json_to_csv(input_dir, output_dir, code_map):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_files = list(input_path.glob('*.json'))
    args = [(json_file, code_map, output_path) for json_file in json_files]

    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap(process_file, args), total=len(json_files)))

if __name__ == '__main__':
    basic_info_path = '/root/autodl-tmp/GoldSparrow/Day_data/qlib_data/basic_info.csv'
#     code,code_name,ipoDate,outDate,type,status
# sh.000001,上证综合指数,1991-07-15,,2,1
# sh.000002,上证A股指数,1992-02-21,,2,1
# sh.000003,上证B股指数,1992-08-17,,2,1

    info_df = pd.read_csv(basic_info_path)  # code,code_name,ipoDate,outDate,type,status
    # 构建映射: 
    # 股票类型: "000002" -> "sh000002"
    # 指数类型: ”000002“ -> "i000002" -> "sh000002"
    code_map = {}
    for row in info_df.itertuples():
        ex, partial = row.code.split('.')
        
        type = str(row.type)
        
        if type == "1": ## 股票类型
            key_code = str(partial)
            val_code = str(ex) + str(partial)
            if key_code in code_map:
                print(f"Duplicate code: {key_code}")
            else:
                code_map[key_code] = val_code
            
            # if partial == "000030":
            #     print(row)
            #     print(f"ex={ex}, partial={partial}")
            #     print(code_map[partial])
            
        elif type == "2": ## 指数类型
            key_code = "i" + str(partial)
            val_code = str(ex) + str(partial)
            if key_code in code_map:
                print(f"Duplicate code: {key_code}")
            else:
                code_map[key_code] = val_code
        else:
            continue


    input_dir = "/root/autodl-tmp/GoldSparrow/Min_data/jvquant"
    output_dir = "/root/autodl-tmp/GoldSparrow/Min_data/Raw"
    
    for year in range(2008, 2025):
        input_files = Path(input_dir) / str(year)
        output_files = Path(output_dir) / str(year)
        convert_json_to_csv(input_files, output_files, code_map)