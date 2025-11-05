import os
import boto3
from urllib.parse import urlparse

# 创建 S3 客户端
s3_client = boto3.client('s3')

# 目标文件夹
output_folder = '/dev/shm/benchmark_img'
os.makedirs(output_folder, exist_ok=True)

# 所有的 txt 文件名（可以根据实际情况进行修改）
txt_files = [f for f in os.listdir() if f.startswith("benchmark_") and f.endswith(".txt")]

# 遍历每个 txt 文件
for txt_file in txt_files:
    local_paths = []  # 本地文件路径列表
    with open(txt_file, 'r') as file:
        s3_paths = file.readlines()
    
    # 处理每个 S3 路径
    for s3_path in s3_paths:
        s3_path = s3_path.strip()  # 去除多余的空格和换行符
        if not s3_path:
            continue
        
        # 解析 S3 路径
        parsed_url = urlparse(s3_path)
        bucket_name = parsed_url.netloc
        key = parsed_url.path.lstrip('/')  # S3 键（路径）

        # 生成本地文件名
        local_filename = os.path.join(output_folder, os.path.basename(key))

        # 下载图片
        s3_client.download_file(bucket_name, key, local_filename)
        local_paths.append(local_filename)  # 保存本地路径
    print(f"Process {s3_path}")

    # 为每个 txt 文件创建对应的 local 路径文件
    local_txt_filename = f"{os.path.splitext(txt_file)[0]}_local.txt"
    with open(local_txt_filename, 'w') as local_file:
        for local_path in local_paths:
            local_file.write(local_path + '\n')

    print(f"{txt_file} 文件的图片下载完成，路径已保存至 {local_txt_filename}")

