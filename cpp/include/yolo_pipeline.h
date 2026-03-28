#pragma once
#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <string>
#include <map>
#include <opencv2/opencv.hpp>
#include "rknn_api.h"

// ================= 配置区 =================
constexpr int MODEL_IN_WIDTH = 640;
constexpr int MODEL_IN_HEIGHT = 640;
constexpr int MODEL_IN_CHANNELS = 6;
constexpr float CONF_THRES = 0.25f;
constexpr float IOU_THRES = 0.45f;
constexpr int NUM_CLASSES = 6;     // 请确认你的实际类别数
constexpr int NUM_ANCHORS = 8400;  // v8 默认输出锚框数
// ==========================================

// 阶段 1：原始任务
struct RawTask { int frame_id; cv::Mat vis_img; cv::Mat ir_img; };

// 阶段 2：预处理完成，等待 NPU 推理
struct NpuTask { 
    int frame_id; 
    cv::Mat orig_vis; 
    cv::Mat orig_ir; 
    std::vector<uint8_t> input_data_6c; 
    float scale; int dw; int dh;
};

// 阶段 3：NPU 推理完成，等待 CPU 解码画框
struct PostTask {
    int frame_id;
    cv::Mat orig_vis;
    cv::Mat orig_ir;
    std::vector<float> box_data;
    std::vector<float> cls_data;
    float scale; int dw; int dh;
};

// 线程安全队列
template <typename T>
class SafeQueue {
private:
    std::queue<T> queue_;
    std::mutex mtx_;
    std::condition_variable cv_;
public:
    void push(T task) {
        std::unique_lock<std::mutex> lock(mtx_);
        queue_.push(task);
        lock.unlock();
        cv_.notify_one();
    }
    bool pop(T& task, bool& is_running) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this, &is_running]() { return !queue_.empty() || !is_running; });
        if (!is_running && queue_.empty()) return false;
        task = queue_.front();
        queue_.pop();
        return true;
    }
};

// 流水线管理器
class PipelineManager {
private:
    SafeQueue<RawTask>  queue_raw_;
    SafeQueue<NpuTask>  queue_npu_;
    SafeQueue<PostTask> queue_post_;
    
    std::vector<std::thread> workers_pre_;
    std::vector<std::thread> workers_npu_;
    std::vector<std::thread> workers_post_;
    
    bool is_running_;
    std::string model_path_;

    cv::VideoWriter video_writer_;        
    std::map<int, cv::Mat> frame_buffer_; 
    int next_write_frame_id_ = 0;        
    std::mutex writer_mtx_;              

    void worker_preprocess();
    void worker_npu_infer(int core_id);
    void worker_postprocess();
    void letterbox(const cv::Mat& src, cv::Mat& dst, float& scale, int& dw, int& dh);
    bool has_stopped_ = false;
public:
    PipelineManager(int num_pre, int num_npu, int num_post, const std::string& model_path);
    ~PipelineManager();
    void push_image(int frame_id, const cv::Mat& vis_img, const cv::Mat& ir_img);
    void stop();
};