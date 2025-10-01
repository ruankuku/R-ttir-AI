import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import imageio

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

MOVEMENT_FILES = {
    "gather": "movement/gather.gif",
    "alarm": "movement/alarm.gif",
    "turnleft": "movement/turnleft.gif",
    "turnright": "movement/turnright.gif",
    "idle": "movement/idle.gif" 
}

class GestureTracker:
    def __init__(self):
        self.gesture_history = deque(maxlen=30) 
        self.last_gesture_time = {"whistle": 0}
        self.current_action = "idle"
        self.action_start_time = time.time()
        self.is_playing = False 

def load_gif(path, target_size=(640, 340)):
    try:
        gif = imageio.mimread(path)
        frames = [cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), target_size) for frame in gif]
        return frames
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return []

ANIMATIONS = {name: load_gif(path) for name, path in MOVEMENT_FILES.items()}

def calculate_finger_distance(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    return ((index_tip.x - thumb_tip.x)**2 + (index_tip.y - thumb_tip.y)**2)**0.5

def calculate_palm_rotation(hand_landmarks):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    
    vector_index = np.array([index_mcp.x - wrist.x, index_mcp.y - wrist.y])
    vector_pinky = np.array([pinky_mcp.x - wrist.x, pinky_mcp.y - wrist.y])
    
    angle = np.degrees(np.arctan2(vector_pinky[1], vector_pinky[0]) - np.arctan2(vector_index[1], vector_index[0]))
    return angle
def detect_gesture(hand_landmarks, tracker):
    current_time = time.time()

    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

    whistle_distance = ((index_tip.x - thumb_tip.x)**2 + (index_tip.y - thumb_tip.y)**2)**0.5
    if whistle_distance < 0.07:  
        if current_time - tracker.last_gesture_time["whistle"] > 3:  
            tracker.last_gesture_time["whistle"] = current_time
            return "alarm"  

    fingers_bent = (
        index_tip.y > index_mcp.y and
        middle_tip.y > middle_mcp.y and
        ring_tip.y > ring_mcp.y and
        pinky_tip.y > pinky_mcp.y
    )
    if fingers_bent:
        return "gather"

    rotation_angle = calculate_palm_rotation(hand_landmarks)
    if rotation_angle > 30:
        return "turnleft"
    elif rotation_angle < -30:
        return "turnright"

    return "idle" 

def main():
    cap = cv2.VideoCapture(0)
    tracker = GestureTracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        current_animation = ANIMATIONS.get(tracker.current_action, ANIMATIONS["idle"])
        if not current_animation:
            print(f"Warning: {tracker.current_action} animation is empty!")
            continue  
        
        if tracker.current_action == "idle":
            frame_index = int((time.time() - tracker.action_start_time) * 10) % len(current_animation)
        elif tracker.is_playing:
            frame_index = int((time.time() - tracker.action_start_time) * 10)
            if frame_index >= len(current_animation):  
                tracker.is_playing = False
                tracker.current_action = "idle"  
                frame_index = 0  
        else:
            frame_index = 0  

        display_frame = current_animation[frame_index].copy()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
            
            gesture = detect_gesture(results.multi_hand_landmarks[0], tracker)
            
            if gesture and gesture != tracker.current_action and not tracker.is_playing:
                tracker.current_action = gesture
                tracker.action_start_time = time.time()
                tracker.is_playing = True  

        cv2.putText(frame, f"Action: {tracker.current_action}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Camera Feed", frame)
        
        if display_frame is not None:
            cv2.imshow("Sheep Herding", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
