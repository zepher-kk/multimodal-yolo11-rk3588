import cv2
import numpy as np
import time
from rknnlite.api import RKNNLite

def preprocess_dummy(img_size=(640, 640)):
    """为了纯测速，我们直接用 Numpy 生成随机矩阵，省去读图时间"""
    # 生成随机的 6 通道浮点矩阵模拟真实的输入
    dummy_6ch = np.random.randint(0, 255, (6, img_size[0], img_size[1]), dtype=np.uint8)
    input_data = np.expand_dims(dummy_6ch, axis=0).astype(np.float32)
    return input_data

def benchmark():
    rknn = RKNNLite()
    print("--> 正在加载模型...")
    rknn.load_rknn('/home/cat/project/rknn/yolo11-mm-rknn/model/yolo11n_mm_rk3588_fp16.rknn') # 或者测你其他的模型
    
    # 默认挂载到一个核心上，如果想榨干性能可以设为 RKNNLite.NPU_CORE_AUTO
    rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

    print("--> 生成 6 通道 Dummy 测速数据...")
    input_data = preprocess_dummy()

    # ================= 预热阶段 (Warm-up) =================
    print("--> 引擎预热中 (10次)...")
    for _ in range(10):
        rknn.inference(inputs=[input_data], data_format=['nchw'])

    # ================= 测速阶段 (Benchmark) =================
    loop_count = 100
    print(f"--> 开始正式测速 (循环 {loop_count} 次)...")
    
    start_time = time.time()
    for _ in range(loop_count):
        rknn.inference(inputs=[input_data], data_format=['nchw'])
    end_time = time.time()

    # ================= 结果统计 =================
    total_time = end_time - start_time
    avg_time_ms = (total_time / loop_count) * 1000.0
    fps = 1.0 / (avg_time_ms / 1000.0)

    print("\n" + "="*40)
    print("🚀 RK3588 NPU 推理速度报告 (纯模型耗时)")
    print(f"模型格式: 6通道输入")
    print(f"测试次数: {loop_count} frames")
    print(f"总计耗时: {total_time:.4f} s")
    print(f"平均单帧耗时: {avg_time_ms:.2f} ms")
    print(f"等效 FPS: {fps:.2f} frames/sec")
    print("="*40 + "\n")

    rknn.release()

if __name__ == '__main__':
    benchmark()