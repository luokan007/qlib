
class my_baostock_data_collector():
    def __init__(self, working_folder="./"):
        self.max_download_limits = 10000 ##baostock一次性下载1万条数据
        self.raw_data_folder_name = "Raw_data"
        self.normalized_data_folder_name = "Normalized_data"
        self.qlib_data_folder_name = "Qlib_data"
        self.working_folder = working_folder
        self.baostock_collector_script_path = "/home/godlike/project/qlib/qlib/scripts/data_collector/baostock_5min/collector.py"
        self.env_name = "BaoStock"
    
    def _activate_conda_cmd(self):
        env_name = self.env_name
        return f'conda activate {env_name}'
    def _generate_baostock_cmd(self,start_date_str=None, end_date_str=None, interval=None,target_folder=None):
        assert(start_date_str & end_date_str & interval & target_folder)
        script_path = self.baostock_collector_script_path
        ret_cmd = f'python {script_path} download_data --source_dir {target_folder} --start {start_date_str} --end {end_date_str} --interval {interval} --region HS300'
        return ret_cmd
    
    def _download_data(self, start_date_str="20080101", end_date_str="20241031", interval=None,target_folder=None):
        ###        
        ###        download data to csv: 
        ###           python collector.py download_data --source_dir /home/godlike/project/GoldSparrow/HighFreq_Data --start 2023-01-01 --end 2024-10-30 --interval 5min --region HS300        

       
        print("downloading data...")
        ##
        set_env_cmd = self._activate_conda_cmd()
        download_cmd = self._generate_baostock_cmd(start_date_str, end_date_str, interval, target_folder)
        full_cmd = f'{set_env_cmd} && {download_cmd}'
 
        ##调用
        with subprocess.Popen(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
            for line in iter(process.stdout.readline, ''):
                print(line.strip())
            ret_code = process.returncode
        
        if ret_code != 0:
            print(f"子进程返回码: {ret_code}")
        
    def get_all_data(self, start_date_str="2008-01-01", end_date_str="2024-10-31"):
        #检查文件夹是否存在，如果没有则创建
        

        ##先下载日频数据，一次性下载
        
        
        ##按照每6个月一批，切割为若干任务，创建5分钟频次数据的任务列表，遍历列表并下载数据
        
        

   
