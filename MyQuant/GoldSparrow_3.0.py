#####################
## high frequency data
## refer: https://github.com/microsoft/qlib/blob/main/scripts/data_collector/baostock_5min/README.md
##  1. download data to csv: 
##      python collector.py download_data --source_dir /home/godlike/project/GoldSparrow/HighFreq_Data --start 2023-01-01 --end 2024-10-30 --interval 5min --region HS300        
##  2. normalize data: 
##      python scripts/data_collector/baostock_5min/collector.py normalize_data
##  3. dump data: 
##      python scripts/dump_bin.py dump_all