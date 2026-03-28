#include "yolo_pipeline.h"
#include <iostream>

PipelineManager::PipelineManager(int num_pre, int num_npu, int num_post, const std::string& model_path)
    : is_running_(true), model_path_(model_path)
{
    // 启动三段式工人
    for (int i = 0; i < num_pre; ++i) workers_pre_.emplace_back(&PipelineManager::worker_preprocess, this);
    for (int i = 0; i < num_npu; ++i) workers_npu_.emplace_back(&PipelineManager::worker_npu_infer, this, i);
    for (int i = 0; i < num_post; ++i) workers_post_.emplace_back(&PipelineManager::worker_postprocess, this);
    std::cout << "✅ 多模态双头异步流水线启动成功！(Pre: " << num_pre << ", NPU: " << num_npu << ", Post: " << num_post << ")\n";
}

PipelineManager::~PipelineManager() {
    stop(); // 析构时安全拉闸
}

// 🌟 新增的核心方法：手动拉闸并等待清空流水线
void PipelineManager::stop() {
    if (has_stopped_) return; // 防止重复执行

    // 优雅退出机制：发送毒丸 (-1) 等待各车间干完活
    for (size_t i = 0; i < workers_pre_.size(); ++i) queue_raw_.push({-1, cv::Mat(), cv::Mat()});
    for (auto& t : workers_pre_) if (t.joinable()) t.join();

    for (size_t i = 0; i < workers_npu_.size(); ++i) queue_npu_.push({-1, cv::Mat(), cv::Mat(), std::vector<uint8_t>(), 0, 0, 0});
    for (auto& t : workers_npu_) if (t.joinable()) t.join();

    for (size_t i = 0; i < workers_post_.size(); ++i) queue_post_.push({-1, cv::Mat(), cv::Mat(), std::vector<float>(), std::vector<float>(), 0, 0, 0});
    for (auto& t : workers_post_) if (t.joinable()) t.join();

    is_running_ = false;
    if (video_writer_.isOpened()) {
        video_writer_.release();
        std::cout << "\n🎬 视频合成完毕！已保存为 result_video.mp4\n";
    }
    has_stopped_ = true;
}

void PipelineManager::push_image(int frame_id, const cv::Mat& vis_img, const cv::Mat& ir_img) {
    queue_raw_.push({frame_id, vis_img.clone(), ir_img.clone()}); 
}

void PipelineManager::letterbox(const cv::Mat& src, cv::Mat& dst, float& scale, int& dw, int& dh) {
    float r = std::min((float)MODEL_IN_WIDTH / src.cols, (float)MODEL_IN_HEIGHT / src.rows);
    int new_unpad_w = std::round(src.cols * r);
    int new_unpad_h = std::round(src.rows * r);
    dw = (MODEL_IN_WIDTH - new_unpad_w) / 2;
    dh = (MODEL_IN_HEIGHT - new_unpad_h) / 2;
    cv::resize(src, dst, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);
    cv::copyMakeBorder(dst, dst, dh, MODEL_IN_HEIGHT - new_unpad_h - dh, dw, MODEL_IN_WIDTH - new_unpad_w - dw, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    scale = r;
}

// ================== 【一阶段：CPU 预处理】 ==================
void PipelineManager::worker_preprocess() {
    RawTask task;
    while (queue_raw_.pop(task, is_running_)) {
        if (task.frame_id == -1) break; 
        
        cv::Mat vis_pad, ir_pad;
        float scale; int dw, dh;
        
        if (task.vis_img.empty()) vis_pad = cv::Mat::zeros(MODEL_IN_HEIGHT, MODEL_IN_WIDTH, CV_8UC3);
        else { cv::cvtColor(task.vis_img, vis_pad, cv::COLOR_BGR2RGB); letterbox(vis_pad, vis_pad, scale, dw, dh); }

        if (task.ir_img.empty()) ir_pad = cv::Mat::zeros(MODEL_IN_HEIGHT, MODEL_IN_WIDTH, CV_8UC3);
        else { cv::cvtColor(task.ir_img, ir_pad, cv::COLOR_BGR2RGB); letterbox(ir_pad, ir_pad, scale, dw, dh); }

        // 6 通道交错拼接 NHWC
        std::vector<uint8_t> input_6c(MODEL_IN_WIDTH * MODEL_IN_HEIGHT * 6);
        for (int i = 0; i < MODEL_IN_HEIGHT * MODEL_IN_WIDTH; ++i) {
            input_6c[i * 6 + 0] = vis_pad.data[i * 3 + 0]; 
            input_6c[i * 6 + 1] = vis_pad.data[i * 3 + 1]; 
            input_6c[i * 6 + 2] = vis_pad.data[i * 3 + 2]; 
            input_6c[i * 6 + 3] = ir_pad.data[i * 3 + 0];  
            input_6c[i * 6 + 4] = ir_pad.data[i * 3 + 1];  
            input_6c[i * 6 + 5] = ir_pad.data[i * 3 + 2];  
        }

        NpuTask npu_task = {task.frame_id, task.vis_img, task.ir_img, std::move(input_6c), scale, dw, dh};
        queue_npu_.push(std::move(npu_task)); 
    }
}

// ================== 【二阶段：NPU 推理】 ==================
void PipelineManager::worker_npu_infer(int core_id) {
    rknn_context ctx = 0;
    rknn_init(&ctx, (void*)model_path_.c_str(), 0, 0, nullptr);
    
    // 绑定核心
    rknn_core_mask mask = RKNN_NPU_CORE_0;
    if (core_id == 1) mask = RKNN_NPU_CORE_1;
    if (core_id == 2) mask = RKNN_NPU_CORE_2;
    rknn_set_core_mask(ctx, mask);

    NpuTask task;
    while (queue_npu_.pop(task, is_running_)) {
        if (task.frame_id == -1) break; 

        rknn_input inputs[1];
        inputs[0].index = 0;
        inputs[0].type = RKNN_TENSOR_UINT8;
        inputs[0].fmt = RKNN_TENSOR_NHWC;
        inputs[0].size = task.input_data_6c.size();
        inputs[0].buf = task.input_data_6c.data();
        rknn_inputs_set(ctx, 1, inputs);

        rknn_run(ctx, nullptr);

        rknn_output outputs[2];
        memset(outputs, 0, sizeof(outputs));
        outputs[0].want_float = 1; outputs[1].want_float = 1;
        rknn_outputs_get(ctx, 2, outputs, nullptr);

        int elements_0 = outputs[0].size / sizeof(float);
        float* out0 = (float*)outputs[0].buf;
        float* out1 = (float*)outputs[1].buf;
        
        float* box_ptr = (elements_0 == NUM_ANCHORS * 4) ? out0 : out1;
        float* cls_ptr = (elements_0 == NUM_ANCHORS * 4) ? out1 : out0;

        std::vector<float> box_data(box_ptr, box_ptr + NUM_ANCHORS * 4);
        std::vector<float> cls_data(cls_ptr, cls_ptr + NUM_ANCHORS * NUM_CLASSES);
        
        rknn_outputs_release(ctx, 2, outputs);

        PostTask post_task = {task.frame_id, task.orig_vis, task.orig_ir, std::move(box_data), std::move(cls_data), task.scale, task.dw, task.dh};
        queue_post_.push(std::move(post_task)); 
    }
    if (ctx > 0) rknn_destroy(ctx);
}

// ================== 【三阶段：CPU 后处理】 ==================
void PipelineManager::worker_postprocess() {
    PostTask task;
    while (queue_post_.pop(task, is_running_)) {
        if (task.frame_id == -1) break; 

        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<int> class_ids;

        // 🌟 解码 C++ 原生连续内存 [channel, 8400]
        for (int i = 0; i < NUM_ANCHORS; ++i) {
            float max_score = 0;
            int max_idx = -1;
            for (int c = 0; c < NUM_CLASSES; ++c) {
                float score = task.cls_data[c * NUM_ANCHORS + i]; 
                if (score > max_score) { max_score = score; max_idx = c; }
            }

            if (max_score > CONF_THRES) {
                float cx = task.box_data[0 * NUM_ANCHORS + i];
                float cy = task.box_data[1 * NUM_ANCHORS + i];
                float w  = task.box_data[2 * NUM_ANCHORS + i];
                float h  = task.box_data[3 * NUM_ANCHORS + i];

                float x1 = ((cx - w / 2.0f) - task.dw) / task.scale;
                float y1 = ((cy - h / 2.0f) - task.dh) / task.scale;
                float x2 = ((cx + w / 2.0f) - task.dw) / task.scale;
                float y2 = ((cy + h / 2.0f) - task.dh) / task.scale;

                boxes.push_back(cv::Rect(x1, y1, x2 - x1, y2 - y1));
                scores.push_back(max_score);
                class_ids.push_back(max_idx);
            }
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, scores, CONF_THRES, IOU_THRES, indices);

        // 优先在可见光上画框
        cv::Mat out_img = task.orig_vis.empty() ? task.orig_ir.clone() : task.orig_vis.clone();
        for (int idx : indices) {
            cv::Rect box = boxes[idx];
            cv::rectangle(out_img, box, cv::Scalar(0, 255, 0), 2);
            std::string label = "Class " + std::to_string(class_ids[idx]) + ": " + std::to_string(scores[idx]).substr(0, 4);
            int baseLine; cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            cv::rectangle(out_img, cv::Rect(cv::Point(box.x, box.y - labelSize.height - 4), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(0, 255, 0), cv::FILLED);
            cv::putText(out_img, label, cv::Point(box.x, box.y - 2), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }

        // 🌟 乱序重排与视频写入
        {
            std::lock_guard<std::mutex> lock(writer_mtx_);
            if (!video_writer_.isOpened()) {
                video_writer_.open("result_video.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30.0, out_img.size());
            }

            frame_buffer_[task.frame_id] = out_img;

            while (frame_buffer_.find(next_write_frame_id_) != frame_buffer_.end()) {
                video_writer_.write(frame_buffer_[next_write_frame_id_]);
                frame_buffer_.erase(next_write_frame_id_);
                next_write_frame_id_++;
            }
        }
        
        static std::mutex cout_mtx;
        std::lock_guard<std::mutex> lock_cout(cout_mtx);
        std::cout << "🚀 第 " << task.frame_id << " 帧处理完工，检出 " << indices.size() << " 个目标。\r" << std::flush;
    }
}