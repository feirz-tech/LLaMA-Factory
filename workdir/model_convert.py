import os
import json
import shutil
from safetensors import safe_open
from safetensors.torch import save_file as safe_save

# ======== 配置部分 ========
input_dir = "checkpoint/KYC-Qwen2.5-VL-72B-Instruct/last_model"   # 原模型目录
output_dir = "checkpoint/KYC-Qwen2.5-VL-72B-Instruct/convert_model"  # 输出目录
os.makedirs(output_dir, exist_ok=True)

# 替换规则
replacements = {
    "model.language_model.": "model.",
    "model.visual.": "visual."
}

# ======== 处理所有 .safetensors 文件 ========
weight_map = {}
total_size = 0

for filename in sorted(os.listdir(input_dir)):
    if not filename.endswith(".safetensors"):
        continue

    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    print(f"🔄 处理 {filename} ...")

    new_tensors = {}

    with safe_open(input_path, framework="pt") as f:
        for key in f.keys():
            new_key = key
            for old, new in replacements.items():
                if new_key.startswith(old):
                    new_key = new_key.replace(old, new, 1)
            tensor = f.get_tensor(key)
            new_tensors[new_key] = tensor

    safe_save(new_tensors, output_path)

    file_size = os.path.getsize(output_path)
    total_size += file_size

    for key in new_tensors.keys():
        weight_map[key] = filename

print("✅ 所有分片已重命名完成。")

# ======== 生成新的 index.json ========
index_data = {
    "metadata": {"total_size": total_size},
    "weight_map": weight_map,
}

index_path = os.path.join(output_dir, "model.safetensors.index.json")
with open(index_path, "w", encoding="utf-8") as f:
    json.dump(index_data, f, indent=2, ensure_ascii=False)

print(f"✅ 新 index 文件已保存到: {index_path}")
print(f"共计 {len(weight_map)} 个权重，模型总大小: {total_size / 1e9:.2f} GB")


src_dir = input_dir
dst_dir = output_dir
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
