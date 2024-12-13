def parse_line(line):
    # 去除行首尾的空白字符（包括换行符）
    line = line.strip()
    
    # 提取时间戳和操作类型
    timestamp, rest = line.split(']: ', 1)
    timestamp = timestamp[3:]  # 去掉开头的中括号,I和空格
    
    # 提取操作类型和股票代码
    action, rest = rest.split(' ', 1)
    stock_code, rest = rest.split(',', 1)
    stock_code = stock_code.strip()
    
    # 将剩余部分按逗号分割成键值对
    fields = {}
    for field in rest.split(','):
        field = field.strip()
        key, value = field.split(' ', 1)
        #print("key",key,"value",value)
        fields[key.strip()] = value.strip()
    
    # 提取所需的字段
    price = float(fields.get('price', 0))
    deal_amount = float(fields.get('deal_amount', 0))
    factor = float(fields.get('factor', 1))  # 默认因子为1，防止除以0
    #print("timestamp",timestamp," action", action, "stock",stock_code, "price",price, deal_amount, factor)
    return timestamp, action, stock_code, price, deal_amount, factor

def calculate_adj_values(price, deal_amount, factor):
    adj_price = price / factor if factor != 0 else price
    adj_deal_amount = deal_amount * factor
    return adj_price, adj_deal_amount

def process_file(input_filename, output_filename):
    with open(input_filename, 'r', encoding='utf-8') as infile, open(output_filename, 'w', encoding='utf-8') as outfile:
        for line in infile:
            line = line.strip()  # 移除行首尾的空白字符（包括换行符）
            if not line:  # 跳过空行
                continue
            try:
                result = parse_line(line)
                if result is not None:
                    timestamp, action, stock_code, price, deal_amount, factor = result
                    adj_price, adj_deal_amount = calculate_adj_values(price, deal_amount, factor)
                    # 将原始日志信息、操作类型以及调整后的数据写入新文件
                    outfile.write(f"[{timestamp}]: {action} {stock_code}, adj_price {adj_price:.6f}, adj_deal_amount {adj_deal_amount:.6f} original price={price}, original deal_amount={deal_amount}, factor={factor}\n")
            except Exception as e:
                # 如果解析失败，保留原始行以供检查
                outfile.write(f"Failed to parse: {line}\nError: {str(e)}\n\n")

# 假设输入文件名为input.txt，输出文件名为output.txt
process_file('qlib_order.txt', 'qlib_order_convert.txt')