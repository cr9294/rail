from ultralytics import YOLO
import cv2
import json

# 加载YOLOv8x模型
model = YOLO("runs/detect/exp2/weights/best.pt")

# 输入视频路径
input_video_path = 'data/part-2.mp4'

# 定义三个检测区域 (x1, y1, x2, y2)
track_1 = (636, 345, 757, 455)
track_2 = (506, 342, 620, 459)
track_3 = (206, 352, 419, 554)

# 初始化存储结果的列表
results_list = []

# 读取输入视频
cap = cv2.VideoCapture(input_video_path)

# 获取视频帧率
frame_fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_id = 0

# 逐帧处理视频
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 视频读取结束

    frame_id += 1  # 每帧的编号

    # 使用YOLO模型进行单帧预测
    results = model.predict(source=frame, save=False, conf=0.5)  # 设置置信度阈值

    # 获取预测结果
    detections = results[0]

    # 初始化当前帧的轨道状态
    track_status = {
        "track_1": "empty",
        "track_2": "empty",
        "track_3": "empty"
    }

    # 遍历检测结果
    for box in detections.boxes:
        # 获取每个边界框的坐标、置信度和类别
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # 转换为整数像素坐标
        confidence = box.conf[0].item()  # 获取置信度
        class_id = int(box.cls[0].item())  # 获取类别ID
        class_name = model.names[class_id]  # 获取类别名称

        # 如果需要指定只检测列车，可以加过滤条件
        if class_name != "train":
            continue

        # 计算边界框中心点坐标
        box_center_x = (x1 + x2) // 2
        box_center_y = (y1 + y2) // 2

        # 检测是否在各轨道区域内
        if track_1[0] <= box_center_x <= track_1[2] and track_1[1] <= box_center_y <= track_1[3]:
            track_status["track_1"] = "occupied"
        if track_2[0] <= box_center_x <= track_2[2] and track_2[1] <= box_center_y <= track_2[3]:
            track_status["track_2"] = "occupied"
        if track_3[0] <= box_center_x <= track_3[2] and track_3[1] <= box_center_y <= track_3[3]:
            track_status["track_3"] = "occupied"

    # 将当前帧的结果存入列表
    results_list.append({
        "frame_id": frame_id,
        "track_status": track_status
    })

# 释放资源
cap.release()

# 将结果保存为 JSON 文件
output_json_path = "output_track_status.json"
with open(output_json_path, "w") as f:
    json.dump(results_list, f, indent=2)

print(f"Detection completed. Results saved to {output_json_path}")