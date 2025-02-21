import datetime
from pathlib import Path
import time
import random
import requests

class MinDataDownloader:
    def __init__(self, url_prefix: str, save_dir: str, stock_list_file: str):
        """
        初始化下载器
        Args:
            url_prefix (str): 下载链接的前缀，例如 "http://data.example.com/min"
            save_dir (str): 数据保存的目录
            stock_list_file (str): 股票代码文件，每行一个股票代码
        """
        self.url_prefix = url_prefix.rstrip('/')
        self.save_dir = Path(save_dir).resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.stock_codes = self._load_stock_codes(stock_list_file)

    def _load_stock_codes(self, stock_list_file: str):
        """
        读取股票代码列表文件
        输入格式：
            sh688408
            sh601601
            sz000569
            sh688314
            sh603378
            sz300982
        """
        codes = []
        with open(stock_list_file, 'r', encoding='utf-8') as f:
            for line in f:
                code = line.strip()
                if code:
                    #sh688408,需要转化为688408
                    if code.startswith("sh"):
                        code = code[2:]
                    elif code.startswith("sz"):
                        code = code[2:]
                    else:
                        print(f"无效的股票代码：{line.strip()}")
                        continue

                    codes.append(code)
        return codes

    def download_all(self):
        """
        根据股票代码以及条件循环构造下载链接，并下载保存目标文件
        这里简单示例，条件循环使用 [1,2,3]，实际可根据需要修改
        """
        #
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        #计算今天到2025-01-01之间有多少天,
        start_date = datetime.datetime(2025, 1, 1)
        delta_days = (start_date - datetime.datetime.now()).days
        #将天数转化为整数
        count = abs(int(delta_days))

        #遍历codes列表，基于prefix生成如：
        # http://121.43.57.182:21936/sql?token=6bed836122a24f2e5fcef1fd85c421c5&mode=minute&code={code}&end_day={today}&limit={count}
        for code in self.stock_codes:
            url = f"{self.url_prefix}&code={code}&end_day={today}&limit={count}"
            # 保存文件，文件名格式：stockcode_today.json
            file_name = f"{code}_{today}.json"
            save_path = self.save_dir / file_name
            time_fmt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            
            # 检查目标文件是否存在，如果存在且大小大于3kb，则跳过该该文件
            if save_path.exists() and save_path.stat().st_size > 3 * 1024:
                print(f"{time_fmt} 文件已存在且大小大于3kb，跳过下载：{save_path}")
                continue
            
            print(f"开始下载：{url}")
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                print(f"{time_fmt} 保存文件：{save_path}")

                # 为确保不被封禁，随机休眠10-20秒
                sleep_duration = random.randint(10, 20)
                time.sleep(sleep_duration)
                
            except Exception as e:
                print(f"下载 {url} 失败：{e}")

        print("所有下载完成！")

if __name__ == '__main__':
    # 示例参数，请根据实际情况修改
    #http://121.43.57.182:21936/sql?token=6bed836122a24f2e5fcef1fd85c421c5&mode=minute&code=600519&end_day=2025-02-19&limit=35
    url_prefix = "http://121.43.57.182:21936/sql?token=6bed836122a24f2e5fcef1fd85c421c5&mode=minute"
    save_dir = "/home/godlike/project/GoldSparrow/Min_Data/dynamic"
    stock_list_file = "/home/godlike/project/GoldSparrow/Day_Data/qlib_data/a_shares_list.txt"

    downloader = MinDataDownloader(url_prefix, save_dir, stock_list_file)
    downloader.download_all()