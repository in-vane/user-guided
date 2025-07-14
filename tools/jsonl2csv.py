import json
import csv

jsonl_file = 'cifar10_results_k2.jsonl'   # 输入文件名
csv_file = 'cifar10_results_k2.csv'      # 输出文件名

# 读取所有行，解析为字典
with open(jsonl_file, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

# 获取所有字段名（假设所有行的字段一致）
fieldnames = data[0].keys()

# 写入 CSV 文件
with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)

print("转换完成！")