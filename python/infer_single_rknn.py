import cv2
import numpy as np
from rknnlite.api import RKNNLite
import argparse

# 你的 6 个类别
CLASSES = ['car', 'truck', 'bus', 'van', 'freight_car', 'people'] 

def preprocess_single(img_path, modality='rgb', img_size=(640, 640)):
    """ 
    单模态预处理: 自动用 0 补齐缺失的 3 个通道
    modality: 'rgb' 或 'ir'
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"图片加载失败: {img_path}")

    orig_h, orig_w = img.shape[:2]

    # 处理有效输入的图片
    img_640 = cv2.resize(img, img_size)
    img_640 = cv2.cvtColor(img_640, cv2.COLOR_BGR2RGB)
    valid_chw = img_640.transpose(2, 0, 1) # (3, 640, 640)
    
    # 创建全 0 的假图用于补位
    zero_chw = np.zeros((3, img_size[0], img_size[1]), dtype=np.uint8)

    # 按 6 通道顺序拼接 (RGB在前，IR在后)

    # --- 修改这段拼接逻辑 (方法三：灰度伪造法) ---
    if modality == 'rgb':
        # 1. 将可见光转为单通道灰度图，剥离色彩信息
        gray = cv2.cvtColor(img_640, cv2.COLOR_RGB2GRAY)
        # 2. 将单通道灰度图重新叠回 3 通道 (因为网络需要 3 个红外通道)
        pseudo_ir_640 = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        pseudo_ir_chw = pseudo_ir_640.transpose(2, 0, 1)
        
        # 3. 拼接: 真实的 RGB + 伪造的灰度 IR
        img_6ch = np.concatenate((valid_chw, pseudo_ir_chw), axis=0)
        
    elif modality == 'ir':
        # 如果只有红外图，情况更好办一点，因为红外本来就没有颜色
        # 把红外图复制给 RGB 通道，相当于网络看到了一张“褪色”的可见光图
        img_6ch = np.concatenate((valid_chw, valid_chw), axis=0)
    
    # 增加 Batch 维度，并强制转为 float32
    input_data = np.expand_dims(img_6ch, axis=0).astype(np.float32)
    
    return input_data, img, orig_w, orig_h

def postprocess(outputs, orig_w, orig_h, conf_thres=0.3, iou_thres=0.45):
    # 和之前一样的 YOLO11 后处理逻辑
    output = outputs[0][0] 
    pred = output.transpose() 
    
    boxes, scores, class_ids = [], [], []
    x_scale, y_scale = orig_w / 640.0, orig_h / 640.0

    for i in range(pred.shape[0]):
        row = pred[i]
        class_scores = row[4:]
        max_score = np.max(class_scores)
        
        if max_score > conf_thres:
            class_id = np.argmax(class_scores)
            cx, cy, w, h = row[:4]
            x_min = int((cx - w / 2) * x_scale)
            y_min = int((cy - h / 2) * y_scale)
            boxes.append([x_min, y_min, int(w * x_scale), int(h * y_scale)])
            scores.append(float(max_score))
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
    return [{"box": boxes[i], "score": scores[i], "class_id": class_ids[i]} for i in indices.flatten()] if len(indices) > 0 else []

def main():
    rknn = RKNNLite()
    
    # 强烈建议使用 FP16 模型，规避刚才的网格框问题
    print("--> 正在加载 RKNN 模型...")
    rknn.load_rknn('/home/cat/project/rknn/yolo11-mm-rknn/model/yolo11n_mm_rk3588_fp16.rknn')
    rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

    # ====== 核心：指定你要测的单图和模态 ======
    test_img_path = '/home/cat/project/rknn/yolo11-mm-rknn/img/vis/00000.png'  # 这里换成你的图
    current_modality = 'rgb'        # 如果测红外，改成 'ir'
    
    print(f"--> 正在准备单模态 ({current_modality}) 输入，自动补零...")
    input_data, orig_img, orig_w, orig_h = preprocess_single(test_img_path, modality=current_modality)

    print("--> 正在 NPU 上推理...")
    # 严正声明 nchw 维度！
    outputs = rknn.inference(inputs=[input_data], data_format=['nchw'])

    results = postprocess(outputs, orig_w, orig_h)

    for res in results:
        x, y, w, h = res['box']
        label = f"{CLASSES[res['class_id']]}: {res['score']:.2f}"
        cv2.rectangle(orig_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(orig_img, label, (x, max(10, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imwrite(f'result_single_{current_modality}.jpg', orig_img)
    print(f"✅ 单模态推理完成！已保存为 result_single_{current_modality}.jpg")
    rknn.release()

if __name__ == '__main__':
    main()