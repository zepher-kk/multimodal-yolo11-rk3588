Multimodal YOLO11 on RK3588 (Visible + Infrared Fusion) 🚀本项目实战演示了如何将多模态（可见光 + 红外）目标检测算法 YOLO11 部署到 RK3588 平台的 NPU 上。采用输入层融合（Early Fusion）架构，通过 6 通道输入实现全天时、全天候的高可靠性目标检测。🌟 核心特性多模态特征互补：结合可见光的高分辨率细节与红外热成像的强抗干扰能力（黑夜、浓雾、逆光）。6通道像素级融合：模型输入端接收 RGB(3) + IR(3) 堆叠产生的 6 通道数据。RK3588 NPU 加速：针对 Rockchip NPU 进行 INT8 量化，充分压榨 6 TOPS 算力。端到端部署流：涵盖从 .pt 模型导出 -> ONNX 简化 -> RKNN 量化转换 -> 板端 Python 高效推理的全流程。📂 项目结构mm_yolov11/
├── model/           # 存放转换后的 .onnx 和 .rknn 模型文件
├── python/          # 核心代码
│   ├── convert.py   # ONNX 转 RKNN 转换脚本
│   └── inference.py # RK3588 板端 Python 推理脚本
├── img/             # 测试图片 (vis/ir)
└── README.md        # 项目说明
🛠️ 部署指南1. 模型训练与导出 (PC端)使用多模态 YOLO11 框架在 M3FD 数据集 上完成训练。执行导出脚本获取 ONNX 模型：建议 Opset = 12必须开启 simplify=True2. 量化数据准备由于模型为 6 通道输入，RKNN 量化校准需要特殊的 .npy 格式数据：# 执行此操作以生成 30 张 6 通道校准数据
python python/gen_calib_data.py 
3. ONNX 转 RKNN在 PC 端（Ubuntu x86）使用 RKNN-Toolkit2 进行转换：python python/convert.py model/best.onnx rk3588 i8 model/yolo11_mm_int8.rknn
4. 板端推理 (RK3588)在鲁班猫等 RK3588 开发板上，使用 RKNN-Toolkit-Lite2 执行推理：python python/inference.py
📊 效果演示可见光 vs 红外特征对比板端检测结果⏱️ 性能基准模型版本运行平台核心数精度延迟 (ms)FPS (端到端)YOLO11n-MMRK3588Core 0INT8~XX ms~XX注：目前使用 Python + OpenCV 预处理，若切换为 C++ + RGA 硬件加速，性能预计可达 40-60 FPS。🔗 模型下载与博客模型权重 (.pt / .rknn): [点击此处下载 (百度网盘/云盘链接)]详细技术博客: 多模态YOLO11算法在RK3588部署实战🤝 交流与贡献如果你对多模态算法或 RKNN 部署感兴趣，欢迎：提交 Issue 讨论问题发送 Pull Request 优化代码觉得本项目有帮助的话，请点一个 Star 🌟！Author: zepher-kkLicense: MIT
