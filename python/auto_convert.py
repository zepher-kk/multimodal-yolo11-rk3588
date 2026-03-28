import os
import subprocess

# ================= 配置区 =================
# 🌟 1. 你的新 ONNX 模型所在的总目录 (WSL环境下的路径)
source_onnx_base_dir = "/mnt/d/learn/dmt/MutilModel_1126/rk3588"

# 🌟 2. 转换后的 RKNN 模型集中保存的总目录 (保持你之前的习惯，方便推理脚本读取)
output_rknn_base_dir = "/mnt/d/learn/rknn/rknn_model_zoo/examples/MM/model"

# RKNN 目标平台
platform = "rk3588"
# 转换脚本名 (确保在运行 auto_convert.py 时，convert.py 在同级目录)
convert_script = "convert.py" 
# ==========================================

def get_version_from_name(name):
    """根据文件夹/模型名称自动推断 YOLO 版本"""
    name_lower = name.lower()
    if "yolov5" in name_lower: return "v5"
    if "yolov8" in name_lower: return "v8"
    if "yolo11" in name_lower: return "v11"
    return None

def main():
    if not os.path.exists(convert_script):
        print(f"❌ 找不到转换脚本 {convert_script}，请确保它在当前运行目录下！")
        return

    total_tasks = 0

    # 获取 source 目录下的所有子文件夹 (例如 yolov5m-mm-early 等)
    folders = [f for f in os.listdir(source_onnx_base_dir) if os.path.isdir(os.path.join(source_onnx_base_dir, f))]

    print(f"🔍 扫描到 {len(folders)} 个模型文件夹，开始流水线作业...\n")

    for folder in folders:
        # 1. 拼接刚刚生成的 ONNX 绝对路径 (对应你 Windows 下的 D:\...\M3FD\weights\xxx.onnx)
        onnx_path = os.path.join(source_onnx_base_dir, folder, "M3FD", "weights", f"{folder}.onnx")
        
        if not os.path.exists(onnx_path):
            print(f"⚠️ 找不到 ONNX 模型 {onnx_path}，跳过...")
            continue

        # 2. 自动判断版本 (v5, v8, v11)
        version = get_version_from_name(folder)
        if not version:
            print(f"⚠️ 无法从名称 {folder} 中识别出 YOLO 版本，跳过...")
            continue

        # 3. 准备集中输出的文件夹路径 (例如: .../model/v5/fp)
        v_dir = os.path.join(output_rknn_base_dir, version)
        fp16_dir = os.path.join(v_dir, "fp")
        i8_dir = os.path.join(v_dir, "i8")

        os.makedirs(fp16_dir, exist_ok=True)
        os.makedirs(i8_dir, exist_ok=True)

        # 4. 配置 RKNN 输出路径
        fp16_rknn_path = os.path.join(fp16_dir, f"{folder}_fp16.rknn")
        i8_rknn_path = os.path.join(i8_dir, f"{folder}_i8.rknn")

        # 5. 执行 FP 转换
        cmd_fp = f"python {convert_script} {onnx_path} {platform} fp {fp16_rknn_path}"
        print(f"\n🚀 [第 {total_tasks + 1} 个] 开始转换 FP16: {folder}")
        subprocess.run(cmd_fp, shell=True)

        # 6. 执行 INT8 转换
        cmd_i8 = f"python {convert_script} {onnx_path} {platform} i8 {i8_rknn_path}"
        print(f"🚀 [第 {total_tasks + 1} 个] 开始转换 INT8: {folder}")
        subprocess.run(cmd_i8, shell=True)
        
        total_tasks += 1

    print("-" * 50)
    print(f"🎉 完美收工！共完成了 {total_tasks} 个模型的双精度转换（共生成 {total_tasks * 2} 个 rknn 文件）。")

if __name__ == '__main__':
    main()