import cv2
import numpy as np
from ultralytics import YOLO

# 加载预训练 YOLO 模型
model = YOLO("runs/detect/exp2/weights/best.pt")  # 确保模型文件路径正确

# 视频路径
video_path = "./data/part-2.mp4"
output_path = "output_video.mp4"

# 设置目标类别（例如：火车的类别为 7）
target_classes = [7]  # COCO 数据集中类别 ID

# 定义感兴趣区域 (ROI) 的顶点
roi_vertices = np.array([[(654, 348), (619, 452), (683, 463), (723, 364), (655, 347)]], dtype=np.int32)

# 计数器
count = 0

def is_inside_roi(point, roi_polygon):
    """判断一个点是否在 ROI 区域内"""
    return cv2.pointPolygonTest(roi_polygon, point, False) >= 0

# 打开视频文件
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 初始化视频保存
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 转换 ROI 顶点为形状 (N, 1, 2)
roi_polygon = roi_vertices.reshape((-1, 1, 2))

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 推理目标检测
    results = model(frame)
    detections = results[0].boxes  # 获取检测框
    overlay = frame.copy()  # 创建叠加图层以绘制 ROI

    # 绘制 ROI 区域
    cv2.polylines(overlay, [roi_polygon], isClosed=True, color=(0, 255, 0), thickness=2)

    # 检查检测结果
    if detections is not None:
        boxes = detections.xyxy.cpu().numpy()  # 检测框坐标
        scores = detections.conf.cpu().numpy()  # 置信度
        classes = detections.cls.cpu().numpy()  # 类别 ID

        for box, score, cls in zip(boxes, scores, classes):
            if cls in target_classes and score > 0.5:  # 筛选目标类别及置信度
                x1, y1, x2, y2 = map(int, box)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # 判断是否在 ROI 区域内
                if is_inside_roi((center_x, center_y), roi_polygon):
                    count += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"Class: {int(cls)}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 在帧上叠加计数信息和 ROI 区域
    alpha = 0.4  # ROI 区域透明度
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, f"Count: {count}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 保存处理后的帧
    out.write(frame)

    # 显示实时处理结果（可选）
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
