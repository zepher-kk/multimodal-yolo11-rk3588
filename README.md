# 🚀 Multi-Modal YOLO on RK3588 (RGB+IR 极限边缘部署)

本项目致力于在边缘计算设备（Rockchip RK3588）上，实现**可见光 (RGB) + 红外光 (IR) 多模态 YOLO 目标检测模型**的高性能落地部署。

针对 RK3588 NPU 的硬件特性，本项目不仅提供了便捷的 Python 验证脚本，更量身打造了 **C++ 三段式异步流水线 (Asynchronous Pipeline)**，彻底榨干 3 颗 NPU 核心的 6 TOPS 算力，实现工业级的极致吞吐量。

## ✨ 核心特性 / Features

- **多模态融合支持**：完美支持 6 通道 (RGB + IR) 图像输入，采用 Early-Fusion / Mid-Fusion 等多种融合策略。
- **全系 YOLO 架构兼容**：自适应兼容 YOLOv5 (无锚框版), YOLOv8, YOLOv11。
- **攻克 DFL 量化坍塌**：针对 YOLOv8/v11 在 NPU 上的 Distribution Focal Loss (DFL) 坍塌问题，提出了源码级的“分离输出头”方案，并采用 FP16 完美保留检测精度。
- **C++ 极限流水线**：利用 `std::thread`、`std::queue` 和 `condition_variable` 构建 **[预处理 CPU] -> [推理 NPU x3] -> [后处理 CPU]** 的异步接力管线，规避全局锁死，帧率跃升 300%！
- **单/双模态自适应**：推理代码支持单模态缺失时的自动全 0 张量补全，提高系统鲁棒性。

------

## 📊 1. 原模型指标 / Model Zoo Metrics

本项目的模型基于 **M3FD** (Multi-Modal Multispectral Fire Detection / or similar) 数据集训练。涵盖了 `n` (Nano) 和 `m` (Medium) 两种参数量级，以及不同的融合阶段。

| **Model Version** |      | **Fusion Strategy** | **Scale** | **RK3588 FP16 FPS (C++)** |
| ----------------- | ---- | ------------------- | --------- | ------------------------- |
| YOLOv11           | i8   | Early-Fusion        | n         | 50.8                      |
| YOLOv11           | i8   | Mid-Fusion          | n         | 36.9                      |
| YOLOv11           | i8   | Early-Fusion        | m         | 22.8                      |
| YOLOv11           | i8   | Mid-Fusion          | m         | 14.5                      |
| YOLOv11           | fp   | Early-Fusion        | n         | 37.7                      |
| YOLOv11           | fp   | Mid-Fusion          | n         | 24.4                      |
| YOLOv11           | fp   | Early-Fusion        | m         | 10.8                      |
| YOLOv11           | fp   | Mid-Fusion          | m         | 6.6                       |
| -                 | -    | -                   | -         | -                         |
| YOLOv8            | i8   | Early-Fusion        | n         | 52.2                      |
| YOLOv8            | i8   | Mid-Fusion          | n         | 43.2                      |
| YOLOv8            | i8   | Early-Fusion        | m         | 26.7                      |
| YOLOv8            | i8   | Mid-Fusion          | m         | 17.2                      |
| YOLOv8            | fp   | Early-Fusion        | n         | 41.1                      |
| YOLOv8            | fp   | Mid-Fusion          | n         | 27.6                      |
| YOLOv8            | fp   | Early-Fusion        | m         | 13.9                      |
| YOLOv8            | fp   | Mid-Fusion          | m         | 8.5                       |
| -                 | -    | -                   | -         | -                         |
| YOLOv5            | i8   | Early-Fusion        | n         | 51.7                      |
| YOLOv5            | i8   | Mid-Fusion          | n         | 41                        |
| YOLOv5            | i8   | Early-Fusion        | m         | 28.3                      |
| YOLOv5            | i8   | Mid-Fusion          | m         | 17.8                      |
| YOLOv5            | fp   | Early-Fusion        | n         | 41.5                      |
| YOLOv5            | fp   | Mid-Fusion          | n         | 27.7                      |
| YOLOv5            | fp   | Early-Fusion        | m         | 14.4                      |
| YOLOv5            | fp   | Mid-Fusion          | m         | 8.7                       |

> **Note:** 上述 FPS 数据是在开启 `-O3` 优化的 C++ 异步流水线架构下，3 核 NPU 满载测得的真实端到端帧率（包含读图、缩放、6通道交织、推理、NMS及写视频）。

------

## ⚙️ 2. 环境与权重配置 / Environment & Weights

### 依赖环境

- **硬件**：Rockchip RK3588 (如 鲁班猫,核心板等)
- **系统**：Ubuntu 20.04 / 22.04 (Linux for ARM64)
- **运行库**：RKNPU2 (`rknn_api.h` 及 `librknnrt.so`), OpenCV 4.x
- **PC 转换端**：RKNN-Toolkit2 (建议在 WSL/Linux x86_64 环境下运行)

### 模型导出与转换秘籍 ⚠️

要在 RK3588 上完美运行最新的 YOLO 模型，**必须分离检测头**，否则会因 Softmax 量化导致输出全乱：

1. 修改 Ultralytics 源码 `head.py` 中的 `Detect.forward()` 方法，在导出时拆开 `box` 和 `cls` 拼接：

   Python

   ```
   if self.export and self.format == "onnx":
       return box, cls  # 斩断 torch.cat 拼接！
   ```

2. 导出 ONNX (推荐 `opset=12`)。

3. 使用 RKNN-Toolkit2 转换为 `FP16` 格式的 `.rknn` 模型。由于 6 通道网络非标，NPU 无法自动转置，请**严格按照 NHWC 格式排布输入内存**。

------

## 🛠️ 3. 使用方法 / Usage

### 🐍 Python 端 (适合快速验证与调试)

Python 脚本内部集成了自动探测坐标格式、张量重排和防乱序机制。

Bash

```
# 1. 进入 python 目录
cd python

# 2. 运行双模态推理测速 (支持预热与循环测试)
python infer_rknn.py \
    -m ../model/v8/fp/yolov8n-mm-early_fp16.rknn \
    -v ../img/vis/00000.png \
    -i ../img/ir/00000.png \
    -ver v8 \
    --warmup 5 --loop 20
```

### ⚡ C++ 端 (工业级极限部署)

采用三段式异步流水线，支持视频流乱序重排与安全回收，压榨极限性能。

Bash

```
# 1. 编译项目
cd cpp
mkdir build && cd build
cmake ..
make -j4  # 多核编译

# 2. 执行流水线推理测试
# 参数: <模型路径> <可见光图像/视频> <红外图像/视频>
./rknn_infer ../../model/v8/fp/yolov8n-mm-early_fp16.rknn ../../img/vis/00000.png ../../img/ir/00000.png
```

*运行结束后，C++ 管线会自动合并绘制好的帧，并在当前目录下生成 `result_video.mp4`。*

------

## 🎯 4. 优化方向 / Future Work

虽然目前 FP16 配合 C++ 流水线已经达成了极高的可用帧率，但项目仍有以下升级空间：

1. **彻底攻克 INT8 DFL 坍塌**：目前的 FP16 规避了精度下降，但并未利用 RK3588 强大的 INT8 算力。下一步计划在 C++ 后处理中纯手写 SIMD 优化的 `Softmax` 积分计算，将 DFL 模块从 NPU 剥离交由 CPU 运算，从而实现无损的 INT8 极致推理。
2. **零拷贝 (Zero-Copy) 图像流**：目前 OpenCV 读取图像及 6 通道内存交织过程仍依赖 CPU。未来可结合 RK3588 的 RGA (2D图形加速引擎) 与 DRM 内存分配，实现从摄像头直通 NPU 的零拷贝预处理。
3. **多目标跟踪 (MOT)**：在流水线的 `Postprocess` 节点后接入 ByteTrack / DeepSORT 算法，实现多模态下的稳定目标追踪。
4. **ROS2 封装集成**：为机器人开发者提供现成的 ROS2 Node 封装，使其可以直接订阅 Sensor 消息并发布 3D/2D Detection Bounding Boxes。

------

### 📝 License & Acknowledgements

- Thanks to [Ultralytics](https://github.com/ultralytics/ultralytics) for the awesome YOLO framework.
- Thanks to the Rockchip RKNPU2 community.
- 如果这个项目帮助到了你的毕业设计或工程落地，欢迎 ⭐ **Star** 鼓励！