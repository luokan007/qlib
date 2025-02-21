from pathlib import Path
from datetime import datetime

def process_files(input_dir, output_dir, threshold_date):
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 转换传入的阈值日期字符串为 datetime 对象
    try:
        date_threshold = datetime.strptime(threshold_date, '%Y-%m-%d')
    except Exception as e:
        print(f"Error parsing threshold_date {threshold_date}: {e}")
        return

    # 获取所有待处理文件的列表
    files = [f for f in input_dir.iterdir() if f.is_file()]

    for file_path in files:
        with file_path.open('r', encoding='utf-8') as f:
            lines = f.readlines()

        original_line_count = len(lines)
        filtered_lines = []

        for line in lines:
            parts = line.split(',')
            if parts:
                try:
                    # 假设日期在第一列，格式为 YYYY-MM-DD
                    line_date = datetime.strptime(parts[0], '%Y-%m-%d')
                    # 只有日期小于或等于阈值的才保留
                    if line_date <= date_threshold:
                        filtered_lines.append(line)
                except Exception:
                    # 无法解析日期则保留该行
                    filtered_lines.append(line)
            else:
                filtered_lines.append(line)

        output_file = output_dir / file_path.name
        with output_file.open('w', encoding='utf-8') as f:
            f.writelines(filtered_lines)
        
        print(f"Processed {file_path.name}: {original_line_count} -> {len(filtered_lines)} lines")

if __name__ == '__main__':
    input_directory = "/home/godlike/project/GoldSparrow/Day_Data/test_raw"
    output_directory = "/home/godlike/project/GoldSparrow/Day_Data/test_raw_ta"
    threshold = "2024-12-30"  # 设置阈值日期，格式为 YYYY-MM-DD
    process_files(input_directory, output_directory, threshold)