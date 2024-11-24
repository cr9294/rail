import cv2

cap = cv2.VideoCapture("./data/part-2.mp4/")
c = 1
frameRate = 10  # 帧数截取间隔（每隔100帧截取一帧）

while True:
    ret, frame = cap.read()
    if ret:
        if c % frameRate == 0:
            print("开始截取视频第：" + str(c) + " 帧")
            # 调整图像尺寸为 640x640
            resized_frame = cv2.resize(frame, (640, 640))
            # 保存调整后的图像
            cv2.imwrite("./data/images/" + str(c) + '.jpg', resized_frame)
        c += 1
        cv2.waitKey(0)
    else:
        print("所有帧都已经保存完成")
        break
cap.release()
