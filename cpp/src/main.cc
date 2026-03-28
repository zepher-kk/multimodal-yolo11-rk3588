//./rknn_infer ../../model/v5/fp/yolov5n-mm-early_fp16.rknn ../../img/vis/00000.png ../../img/ir/00000.png
#include <iostream>
#include <chrono>
#include "yolo_pipeline.h"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <model.rknn> <vis_img> <ir_img>\n";
        return -1;
    }

    std::string model_path = argv[1];
    
    // 初始化流水线 (这个过程可能需要几秒钟加载 NPU 模型，我们不把它算进测速里)
    PipelineManager pipeline(2, 3, 2, model_path);

    cv::Mat vis_img = cv::imread(argv[2]);
    cv::Mat ir_img  = cv::imread(argv[3]);

    if (vis_img.empty() && ir_img.empty()) {
        std::cerr << "❌ 无法读取图像！" << std::endl;
        return -1;
    }

    int test_frames = 300; 
    std::cout << "🚀 开始以最高性能流水线模式灌入 " << test_frames << " 帧数据...\n";

    // 🌟 真正开始掐表测速！
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < test_frames; ++i) {
        pipeline.push_image(i, vis_img, ir_img);
    }

    std::cout << "📥 数据已全部进入管线，正在全速处理...\n";
    
    // 🌟 阻塞主线程，直到所有帧被画完框并存入视频
    pipeline.stop(); 

    // 🌟 掐表结束
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;
    
    // 计算专业性能指标
    double total_time_ms = duration.count();
    double avg_time_ms = total_time_ms / test_frames;
    double fps = 1000.0 / avg_time_ms;

    std::cout << "\n========================================\n";
    std::cout << "📊 流水线极限测速报告 (共 " << test_frames << " 帧)\n";
    std::cout << "⏱️ 总耗时:      " << total_time_ms << " ms\n";
    std::cout << "⚡ 单帧平均耗时: " << avg_time_ms << " ms\n";
    std::cout << "🚀 极限吞吐量:   " << fps << " FPS\n";
    std::cout << "========================================\n";

    return 0; 
}