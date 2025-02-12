from pathlib import Path

def process_files(input_dir, output_dir):
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 如果输入和输出目录相同，先获取所有待处理文件的列表
    files = [f for f in input_dir.iterdir() if f.is_file()]

    for file_path in files:
        with file_path.open('r', encoding='utf-8') as f:
            lines = f.readlines()

        original_line_count = len(lines)
        # 如果行数大于 10，则删除最后 10 行
        if original_line_count > 10:
            lines = lines[:-10]

        # 如果输入和输出目录相同，则直接覆盖原文件
        output_file = output_dir / file_path.name
        with output_file.open('w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print(f"Processed {file_path.name}: {original_line_count} -> {len(lines)} lines")

if __name__ == '__main__':

    input_directory = "/home/godlike/project/GoldSparrow/Day_Data/test_raw_ta"
    output_directory = "/home/godlike/project/GoldSparrow/Day_Data/test_raw_ta"
    process_files(input_directory, output_directory)