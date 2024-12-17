def divide_number_evenly(total, parts, multiple=100):
    if parts <= 0:
        raise ValueError("The number of parts must be greater than 0.")
    
    # 确保total是multiple的倍数
    if total % multiple != 0:
        raise ValueError(f"The total amount {total} is not a multiple of {multiple}.")
    
    # 计算每份的基础值，确保它是multiple的倍数
    base_value = (total // parts) // multiple * multiple
    
    # 计算剩余量
    remainder = total - base_value * parts
    
    # 如果有剩余量，需要将其以multiple为单位分配给前面的份额
    extra_parts = remainder // multiple

    result = []
    for i in range(parts):
        if i < extra_parts:
            result.append(base_value + multiple)
        else:
            result.append(base_value)
    
    # 检查结果是否正确
    if sum(result) != total:
        raise ValueError("The sum of the divided parts does not match the total.")

    return result

# 使用示例
total = 3000
parts = 8
print(divide_number_evenly(total, parts))