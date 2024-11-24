# 原始 YOLO 格式数据
yolo_data = [
    [1, 0.544531, 0.555556, 0.095312, 0.152778],
    [2, 0.439844, 0.556250, 0.089063, 0.162500],
    [0, 0.244531, 0.629861, 0.167187, 0.281944]
]

# 图像的宽度和高度
image_width = 1280
image_height = 720

# 转换 YOLO 格式为像素坐标
converted_data = []
for item in yolo_data:
    class_id, x_center, y_center, width, height = item

    # 转换为像素值
    x_center_pixel = x_center * image_width
    y_center_pixel = y_center * image_height
    width_pixel = width * image_width
    height_pixel = height * image_height

    # 计算左上角和右下角坐标
    x1 = int(x_center_pixel - width_pixel / 2)
    y1 = int(y_center_pixel - height_pixel / 2)
    x2 = int(x_center_pixel + width_pixel / 2)
    y2 = int(y_center_pixel + height_pixel / 2)

    # 保存转换后的数据
    converted_data.append((x1, y1, x2, y2))

# 格式化输出为检测区域的格式
print("# 定义三个检测区域 (x1, y1, x2, y2)")
for i, (x1, y1, x2, y2) in enumerate(converted_data, start=1):
    print(f"track_{i} = ({x1}, {y1}, {x2}, {y2})")