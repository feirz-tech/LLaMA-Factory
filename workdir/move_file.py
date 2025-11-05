import os
import shutil

src_dir = "checkpoint/KYC-Qwen2.5-VL-72B-Instruct_clean_data/last_model"            # 原目录
dst_dir = "checkpoint/KYC-Qwen2.5-VL-72B-Instruct_clean_data/convert_model"    # 目标目录

os.makedirs(dst_dir, exist_ok=True)

for root, dirs, files in os.walk(src_dir):
    # 计算相对路径
    rel_path = os.path.relpath(root, src_dir)
    dst_subdir = os.path.join(dst_dir, rel_path)
    os.makedirs(dst_subdir, exist_ok=True)

    for file in files:
        if file == "model.safetensors.index.json":
            continue
        if file.endswith(".safetensors"):
            continue

        src_path = os.path.join(root, file)
        dst_path = os.path.join(dst_subdir, file)

        shutil.copy2(src_path, dst_path)

print("✅ 目录拷贝完成（已排除 safetensors 文件和 index.json）")
