import cv2
import numpy as np
from rknnlite.api import RKNNLite

# YOLO11 类别列表 (M3FD 的 6 个类别，按你之前 yaml 的顺序)
CLASSES = ['car', 'truck', 'bus', 'van', 'freight_car', 'people'] 

def preprocess(rgb_path, ir_path, img_size=(640, 640)):
    """ 
    6通道多模态预处理: 分别读取、Resize、转RGB、拼接到一起
    """
    img_rgb = cv2.imread(rgb_path)
    img_ir = cv2.imread(ir_path)
    
    if img_rgb is None or img_ir is None:
        raise ValueError("图片加载失败，请检查路径！")

    # 保存原始尺寸用于后续画框还原
    orig_h, orig_w = img_rgb.shape[:2]

    # 1. Resize (这里采用最简单的直接缩放，与你的校准脚本保持一致)
    rgb_640 = cv2.resize(img_rgb, img_size)
    ir_640 = cv2.resize(img_ir, img_size)
    
    # 2. BGR 转 RGB
    rgb_640 = cv2.cvtColor(rgb_640, cv2.COLOR_BGR2RGB)
    ir_640 = cv2.cvtColor(ir_640, cv2.COLOR_BGR2RGB)
    
    # 3. 维度转换 HWC -> CHW (3, 640, 640)
    rgb_chw = rgb_640.transpose(2, 0, 1)
    ir_chw = ir_640.transpose(2, 0, 1)
    
    # 4. 在通道维度拼接为 6 通道 (6, 640, 640)
    img_6ch = np.concatenate((rgb_chw, ir_chw), axis=0)
    
    # 5. 增加 Batch 维度 (1, 6, 640, 640)
    input_data = np.expand_dims(img_6ch, axis=0)
    
    return input_data, img_rgb, orig_w, orig_h

def postprocess(outputs, orig_w, orig_h, conf_thres=0.3, iou_thres=0.45):
    """
    YOLO11 后处理: 解析 (1, 84, 8400) 矩阵并进行 NMS
    """
    # RKNN 返回的是 list，取出第一个 Tensor，并压缩 Batch 维度
    # output 形状变为 (84, 8400)
    output = outputs[0][0] 
    
    # 转置为 (8400, 84) -> 8400 个预测框，84 = 4 (cx,cy,w,h) + 80 (或者你的6个) 类别置信度
    pred = output.transpose() 
    
    boxes = []
    scores = []
    class_ids = []
    
    # 计算缩放比例
    x_scale = orig_w / 640.0
    y_scale = orig_h / 640.0

    for i in range(pred.shape[0]):
        row = pred[i]
        # 类别置信度从第 4 个元素开始
        class_scores = row[4:]
        max_score = np.max(class_scores)
        
        if max_score > conf_thres:
            class_id = np.argmax(class_scores)
            
            # 解析中心点和宽高
            cx, cy, w, h = row[:4]
            
            # 还原到原图尺寸上的坐标 (OpenCV NMS 需要格式为 [x_min, y_min, width, height])
            x_min = int((cx - w / 2) * x_scale)
            y_min = int((cy - h / 2) * y_scale)
            box_w = int(w * x_scale)
            box_h = int(h * y_scale)
            
            boxes.append([x_min, y_min, box_w, box_h])
            scores.append(float(max_score))
            class_ids.append(class_id)

    # 执行 OpenCV 自带的 NMS (非极大值抑制)
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
    
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            results.append({
                "box": boxes[i], 
                "score": scores[i], 
                "class_id": class_ids[i]
            })
    return results

def main():
    # 1. 实例化 RKNNLite
    rknn = RKNNLite()

    # 2. 载入模型 (替换为你的 rknn 路径)
    print("--> 正在加载 RKNN 模型...")
    ret = rknn.load_rknn('/home/cat/project/rknn/yolo11-mm-rknn/model/yolo11n_mm_rk3588_i8.rknn')
    if ret != 0:
        print("加载模型失败！")
        return

    # 3. 初始化运行环境 (RK3588 有三个 NPU 核心，这里指定挂载到 Core 0)
    print("--> 初始化 NPU 运行环境...")
    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    if ret != 0:
        print("初始化运行环境失败！")
        return

    # 4. 准备数据
    print("--> 正在准备 6 通道输入数据...")
    rgb_path = '/home/cat/project/rknn/yolo11-mm-rknn/img/vis/00000.png'  # 替换为你板子上的图片路径
    ir_path = '/home/cat/project/rknn/yolo11-mm-rknn/img/ir/00000.png'    # 替换为你板子上的图片路径
    input_data, orig_img, orig_w, orig_h = preprocess(rgb_path, ir_path)

    # 5. 执行 NPU 推理
    # 5. 执行 NPU 推理
    print("--> 正在 NPU 上狂飙推理...")
    # 强制将 uint8 转为 float32，并严正声明这是 nchw 格式，禁止 RKNN 瞎转置！
    input_data = input_data.astype(np.float32)
    outputs = rknn.inference(inputs=[input_data], data_format=['nchw'])

    # 6. 后处理解码
    print("--> 执行 YOLO11 矩阵解码与 NMS...")
    results = postprocess(outputs, orig_w, orig_h, conf_thres=0.3, iou_thres=0.45)

    # 7. 绘制结果
    for res in results:
        x, y, w, h = res['box']
        score = res['score']
        class_id = res['class_id']
        label = f"{CLASSES[class_id]}: {score:.2f}"
        
        # 画框和写字
        cv2.rectangle(orig_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(orig_img, label, (x, max(10, y - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 保存结果
    cv2.imwrite('result_output_i8.jpg', orig_img)
    print("✅ 大功告成！结果已保存至 result_output.jpg，快去看看吧！")

    # 释放资源
    rknn.release()

if __name__ == '__main__':
    main()