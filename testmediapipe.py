import cv2
import mediapipe as mp

# ✅ 初始化 MediaPipe 手部检测
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# ✅ 打开摄像头
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 将 BGR 图像转换为 RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 进行手势检测
    results = hands.process(rgb_frame)

    # 如果检测到手部
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 画出手部关键点
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 显示摄像头画面
    cv2.imshow("MediaPipe Hand Detection", frame)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
