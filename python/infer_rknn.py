'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''' 
#1. 完整双模态测速（默认预热 5 次，测试 20 次）：
#Bash
#python python/infer_rknn.py -m model/v8/i8/yolov8n-mm-early_i8.rknn -v img/vis/00000.png -i img/ir/00000.png -ver v8
#2. 仅用红外图推理（验证模型的鲁棒性）：
#Bash
#python python/infer_rknn.py -m model/v8/i8/yolov8n-mm-early_i8.rknn -i img/ir/00000.png
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import argparse
import cv2
import numpy as np
import time
import os

# 智能兼容 PC 端仿真与开发板端真实推理
try:
    from rknnlite.api import RKNNLite as RKNN
    print("[INFO] 检测到 rknnlite，正运行在开发板环境")
except ImportError:
    from rknn.api import RKNN
    print("[INFO] 检测到 rknn.api，正运行在 PC 仿真环境")

def parse_args():
    parser = argparse.ArgumentParser(description="高效多模态 YOLO RKNN 推理与测速脚本 🚀")
    parser.add_argument("-m", "--model", type=str, required=True, help="RKNN 模型文件路径")
    parser.add_argument("-v", "--vis", type=str, default=None, help="可见光 (Visible) 图像路径 (可选)")
    parser.add_argument("-i", "--ir", type=str, default=None, help="红外光 (Infrared) 图像路径 (可选)")
    parser.add_argument("-ver", "--version", type=str, choices=["v5", "v8", "v11"], required=True, help="YOLO 模型版本")
    parser.add_argument("-s", "--imgsz", type=int, default=640, help="推理尺寸，默认 640")
    parser.add_argument("-c", "--conf_thres", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("-iou", "--iou_thres", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("-o", "--out_dir", type=str, default="result", help="输出文件夹路径")
    parser.add_argument("--warmup", type=int, default=1, help="NPU 预热次数 (默认 1)")
    parser.add_argument("--loop", type=int, default=1, help="测速循环次数 (默认 1)")
    
    args = parser.parse_args()
    if args.vis is None and args.ir is None:
        parser.error("❌ 必须至少提供一种输入图像 (-v/--vis 或 -i/--ir)")
    return args

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """ 强制等比例缩放并严格填充到指定尺寸 (专治 NPU 静态输入) """
    shape = img.shape[:2]  # current shape [height, width]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    
    dw, dh = dw / 2, dh / 2  
    
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def nms(boxes, scores, iou_thres):
    """ 纯 NumPy 实现的 Non-Maximum Suppression """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep

def postprocess(outputs, version, conf_thres, iou_thres):
    out0 = np.squeeze(outputs[0])
    out1 = np.squeeze(outputs[1])
    
    if len(out0.shape) == 2 and out0.shape[0] < out0.shape[1]:
        out0 = np.transpose(out0, (1, 0))  
        out1 = np.transpose(out1, (1, 0))
        
    if out0.shape[1] == 4:
        box_out, cls_out = out0, out1
    else:
        box_out, cls_out = out1, out0
        
    boxes_cxcywh = box_out
    
    if cls_out.shape[1] == 6:
        cls_scores = cls_out
        class_ids = np.argmax(cls_scores, axis=1)
        scores = np.max(cls_scores, axis=1)
    else:
        obj_conf = cls_out[:, 0]
        cls_probs = cls_out[:, 1:]
        class_ids = np.argmax(cls_probs, axis=1)
        scores = obj_conf * np.max(cls_probs, axis=1)

    valid_mask = scores > conf_thres
    boxes_cxcywh = boxes_cxcywh[valid_mask]
    scores = scores[valid_mask]
    class_ids = class_ids[valid_mask]

    if len(boxes_cxcywh) == 0:
        return np.array([]), np.array([]), np.array([])

    boxes_xyxy = np.empty_like(boxes_cxcywh)
    boxes_xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
    
    keep_indices = nms(boxes_xyxy, scores, iou_thres)
    
    return boxes_xyxy[keep_indices], scores[keep_indices], class_ids[keep_indices]

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    orig_img = None
    r, dw, dh = 1.0, 0, 0
    
    # 🌟 核心逻辑 1：智能单/双模态读取与全 0 补全
    if args.vis:
        vis_img = cv2.imread(args.vis)
        if vis_img is None: print("❌ 读取可见光图像失败！"); return
        orig_img = vis_img.copy()  # 默认在可见光图上画框
        vis_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        vis_pad, r, (dw, dh) = letterbox(vis_rgb, new_shape=(args.imgsz, args.imgsz))
    else:
        # 缺失可见光，用全 0 纯黑张量代替
        vis_pad = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)

    if args.ir:
        ir_img = cv2.imread(args.ir)
        if ir_img is None: print("❌ 读取红外图像失败！"); return
        if orig_img is None: 
            orig_img = ir_img.copy() # 如果没提供可见光，就在红外图上画框
        ir_rgb = cv2.cvtColor(ir_img, cv2.COLOR_BGR2RGB)
        ir_pad, r, (dw, dh) = letterbox(ir_rgb, new_shape=(args.imgsz, args.imgsz))
    else:
        # 缺失红外光，用全 0 纯黑张量代替
        ir_pad = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    
    # 组合为 6 通道
    img_6c = np.concatenate((vis_pad, ir_pad), axis=2)
    
    # 🌟 核心修复：坚决不能加 transpose！RKNN Python API 必须吃 NHWC 格式，内部会自动转换！
    img_6c = np.expand_dims(img_6c, axis=0) # -> (1, 640, 640, 6)

    print(f"🚀 正在初始化 RKNN...")
    rknn = RKNN()
    if rknn.load_rknn(args.model) != 0: return
    if rknn.init_runtime() != 0: return

    # 🌟 核心逻辑 2：标准测速流程 (预热 + 多次推理)
    print(f"🔥 开始 NPU 预热 ({args.warmup} 次)...")
    for _ in range(args.warmup):
        rknn.inference(inputs=[img_6c])
        
    print(f"⏱️ 开始测速 (连续推理 {args.loop} 次)...")
    t0 = time.time()
    for _ in range(args.loop):
        outputs = rknn.inference(inputs=[img_6c]) 
    t1 = time.time()
    
    avg_time_ms = ((t1 - t0) * 1000) / args.loop
    fps = 1000.0 / avg_time_ms
    print(f"✅ 测速完成！平均耗时: {avg_time_ms:.2f} ms | 帧率 (FPS): {fps:.2f}")

    # 使用最后一次的 output 进行后处理和画框
    boxes, scores, class_ids = postprocess(outputs, args.version, args.conf_thres, args.iou_thres)
    
    if len(boxes) > 0:
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / r
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / r
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_img.shape[1])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_img.shape[0])
    
        for box, score, cid in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Class {cid}: {score:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(orig_img, (x1, y1 - h - 4), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(orig_img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    model_basename = os.path.basename(args.model).replace('.rknn', '')
    mode_str = "dual" if (args.vis and args.ir) else ("vis_only" if args.vis else "ir_only")
    timestamp = time.strftime("%H%M%S")
    out_path = os.path.join(args.out_dir, f"{model_basename}_{mode_str}_{timestamp}.jpg")
    
    cv2.imwrite(out_path, orig_img)
    print(f"🎨 渲染结果已保存至: {out_path}")
    
    rknn.release()

if __name__ == '__main__':
    main()