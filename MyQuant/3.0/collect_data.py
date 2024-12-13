#####################
## baostock数据下载的流程
## refer: https://github.com/microsoft/qlib/blob/main/scripts/data_collector/baostock_5min/README.md
##  1. download data to csv: 
##      python collector.py download_data --source_dir /home/godlike/project/GoldSparrow/HighFreq_Data --start 2023-01-01 --end 2024-10-30 --interval 5min --region HS300        
##  2. normalize data: 
##      python collector.py normalize_data --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data --source_dir ~/.qlib/stock_data/source/hs300_5min_original --normalize_dir ~/.qlib/stock_data/source/hs300_5min_nor --region HS300 --interval 5min
##  3. dump data: 
##      python dump_bin.py dump_all --csv_path ~/.qlib/stock_data/source/hs300_5min_nor --qlib_dir ~/.qlib/qlib_data/hs300_5min_bin --freq 5min --exclude_fields date,symbol
##
##  项目目标：
##    - 构建一次性数据下载
##    - 构建日频数据更新
##  一次性下载设计思想：
##    - 使用qlib自带collector下载数据，分别下载1d数据和5min数据，分成两个任务下载
##    - baostock下载的数据目前看上限为10000条，5min数据需要切割成多个任务并存放到临时文件夹
##    - 将多个任务的csv文件进行拼接
##    - 使用qlib的 collector脚本完成normalize，并用dump_bin脚本生成qlib的存储格式
##  持续更新设计思想：
##    - 创建crontab任务，并运行脚本
##    - 获取当日日期，使用qlib自带collector下载数据，分别下载1d数据和5min数据，分成两个任务下载
##    - 与旧数据拼接
##    - normalized data
##    - dump data
##  项目目录结构：
##  Raw_data： 下载的原始文件夹
##    |-- dowload_1d_data: 下载1d数据的临时文件夹夹
##    |----[startdate]_[enddate]：起止日的文件夹
##    |-----------[stock_code].csv：股票数据，
##    |-- download_5min_data: 下载5min数据的临时文件夹
##    |----[startdate]_[enddate]：起止日的文件夹
##    |-----------[stock_code].csv：股票数据，
##  Normalized_data： normalized数据
##    |--- 5min_data: 5min数据
##    |--- 1day_data
##  Qlib_data： qlib存储格式
############################

import os
import subprocess
from datetime import datetime, timedelta

class my_baostock_data_collector():
    def __init__(self, working_folder="./"):
        self.max_download_limits = 10000  # Baostock一次性下载1万条数据
        self.raw_data_folder_name = "Raw_data"
        self.normalized_data_folder_name = "Normalized_data"
        self.qlib_data_folder_name = "Qlib_data"
        self.working_folder = working_folder
        self.baostock_collector_script_path = "/home/godlike/project/qlib/qlib/scripts/data_collector/baostock_5min/collector.py"
        self.dump_bin_script_path = "/home/godlike/project/qlib/qlib/scripts/dump_bin.py"
        self.qlib_1day_dir = "/home/godlike/project/GoldSparrow/Updated_Stock_Data"
        self.env_name = "BaoStock"

    def _activate_conda_cmd(self):
        """生成激活 Conda 环境的命令"""
        env_name = self.env_name
        return f'. /home/godlike/miniconda3/etc/profile.d/conda.sh && conda activate {env_name}'

    def _generate_baostock_cmd(self, start_date_str=None, end_date_str=None, interval=None, target_folder=None):
        """生成从 Baostock 下载数据的命令"""
        assert start_date_str and end_date_str and interval and target_folder
        script_path = self.baostock_collector_script_path
        ret_cmd = (
            f'python {script_path} download_data '
            f'--source_dir {target_folder} '
            f'--start {start_date_str} --end {end_date_str} '
            f'--interval {interval} --region HS300'
        )
        return ret_cmd

    def _generate_normalize_cmd(self, source_dir=None, normalize_dir=None, freq=None,qlib_data_1d_dir=None):
        """生成归一化数据的命令"""
        assert source_dir and normalize_dir and freq
        if freq == "1d":
            ret_cmd = (
                f'python {self.baostock_collector_script_path} normalize_data '
                f'--source_dir {source_dir} '
                f'--normalize_dir {normalize_dir} '
                f'--region HS300 --interval {freq}'
            )
            return ret_cmd
        elif freq == "5min":
            assert qlib_data_1d_dir
            ret_cmd = (
                f'python {self.baostock_collector_script_path} normalize_data '
                f'--qlib_data_1d_dir {qlib_data_1d_dir} '
                f'--source_dir {source_dir} '
                f'--normalize_dir {normalize_dir} '
                f'--region HS300 --interval {freq}'
            )
            return ret_cmd
        else:
            raise ValueError("Invalid frequency")

    def _generate_dump_cmd(self, csv_path=None, qlib_dir=None, freq=None):
        """生成将数据转储为 Qlib 格式的命令"""
        assert csv_path and qlib_dir and freq
        ret_cmd = (
            f'python {self.dump_bin_script_path} dump_all '
            f'--csv_path {csv_path} '
            f'--qlib_dir {qlib_dir} '
            f'--freq {freq} '
            f'--exclude_fields date,symbol'
        )
        return ret_cmd

    def _create_folders_if_not_exist(self, *folders):
        """创建不存在的文件夹"""
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)

    def _download_data(self, start_date_str, end_date_str, interval, target_folder):
        """下载指定时间段和间隔的数据"""
        print("downloading data...")
        set_env_cmd = self._activate_conda_cmd()
        download_cmd = self._generate_baostock_cmd(start_date_str, end_date_str, interval, target_folder)
        full_cmd = f'{set_env_cmd} && {download_cmd}'
        #full_cmd = f'{download_cmd}'
        print(f'running command: {full_cmd}')

        with subprocess.Popen(full_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
            for line in iter(process.stdout.readline, ''):
                print(line.strip())
            ret_code = process.wait()

        if ret_code != 0:
            print(f"子进程返回码: {ret_code}")

    def _merge_stock_csv_files(self, stock_file_path, merged_5min_folder, stock_code, is_new = False):
        """合并单个股票的所有 CSV 文件，并去掉第二个及以后文件的第一行"""
        output_file_path = os.path.join(merged_5min_folder, f"{stock_code}.csv")
        if is_new:
            ##是一个新的文件，创建csv文件
            with open(stock_file_path, 'r') as infile:
                with open(output_file_path, 'w') as outfile:
                    outfile.write(infile.read())
        else:
            ##是一个已有的文件，直接追加，忽略第一行
            with open(stock_file_path, 'r') as infile:
                with open(output_file_path, 'a') as outfile:
                    lines = infile.readlines()[1:]  # Skip the header
                    outfile.writelines(lines)           

    def get_all_data(self, start_date_str="2008-01-01", end_date_str="2024-10-31"):
        """主函数：获取所有数据并处理"""
        raw_data_folder = os.path.join(self.working_folder, self.raw_data_folder_name)
        normalized_data_folder = os.path.join(self.working_folder, self.normalized_data_folder_name)
        qlib_data_folder = os.path.join(self.working_folder, self.qlib_data_folder_name)

        download_5min_data_folder = os.path.join(raw_data_folder, "download_5min_data")
        merged_5min_folder = os.path.join(raw_data_folder, "Merged_5min")

        self._create_folders_if_not_exist(download_5min_data_folder, merged_5min_folder)

        # Download 5-minute data in batches
        print("Downloading 5-minute data in batches...")
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        
        qlib_1day_dir = self.qlib_1day_dir

        while start_date < end_date:
            batch_end_date = min(start_date + timedelta(days=180), end_date)  # 每次处理6个月的数据
            batch_start_str = start_date.strftime("%Y-%m-%d")
            batch_end_str = batch_end_date.strftime("%Y-%m-%d")
            batch_target_folder = os.path.join(download_5min_data_folder, f"{batch_start_str}_{batch_end_str}")
            self._create_folders_if_not_exist(batch_target_folder)
            self._download_data(batch_start_str, batch_end_str, "5min", batch_target_folder)
            start_date = batch_end_date + timedelta(days=1)

        # Merge 5-minute CSV files by stock code
        tmp_stock_dic = {}
        print("Merging 5-minute CSV files by stock code...")
        for stock_batch_folder in os.listdir(download_5min_data_folder):
            stock_folder_path = os.path.join(download_5min_data_folder, stock_batch_folder)
            if os.path.isdir(stock_folder_path):
                for filename in os.listdir(stock_folder_path):
                    if filename.endswith('.csv'):
                        stock_file_path = os.path.join(stock_folder_path, filename)
                        stock_code = filename.split('.')[0]
                        
                        ##如果stock_code不在字典中，则添加到字典中，调用_merge_stock_csv_files函数，参数is_new 为True
                        if stock_code not in tmp_stock_dic:
                            tmp_stock_dic[stock_code] = True
                            self._merge_stock_csv_files(stock_file_path, merged_5min_folder,stock_code, True)
                        else:
                            self._merge_stock_csv_files(stock_file_path, merged_5min_folder,stock_code, False)

        # Normalize and dump 5-minute data
        print("Normalizing and dumping 5-minute data...")
        normalize_5min_dir = os.path.join(normalized_data_folder, "5min_data")
        qlib_5min_dir = os.path.join(qlib_data_folder, "hs300_5min_bin")
        self._create_folders_if_not_exist(normalize_5min_dir, qlib_5min_dir)
        temp_normalized_source = merged_5min_folder
        
        normalize_5min_cmd = self._generate_normalize_cmd(temp_normalized_source, normalize_5min_dir, "5min",qlib_1day_dir)
        dump_5min_cmd = self._generate_dump_cmd(normalize_5min_dir, qlib_5min_dir, "5min")
        self._run_commands([normalize_5min_cmd, dump_5min_cmd])

    def _run_commands(self, commands):
        """运行一系列命令"""
        for cmd in commands:
            print(f"Running command: {cmd}")
            set_env_cmd = self._activate_conda_cmd()
            #
            full_cmd = f'{set_env_cmd} && {cmd}'

            with subprocess.Popen(full_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
                for line in iter(process.stdout.readline, ''):
                    print(line.strip())
                ret_code = process.wait()

            if ret_code != 0:
                print(f"Command failed with return code: {ret_code}")

# Example usage:
if __name__ == "__main__":
    collector = my_baostock_data_collector("/home/godlike/project/GoldSparrow/HighFreq_Data/")
    collector.get_all_data("2023-01-01", "2024-10-31")



