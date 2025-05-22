import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

#Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence = 0.5,
                       min_tracking_confidence = 0.5)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # or use a video file e.g. 'jumpshot.mp4'

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Example: get y-position of left hip
        h, w, _ = frame.shape
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        cx, cy = int(left_hip.x * w), int(left_hip.y * h)
        cv2.circle(frame, (cx, cy), 8, (255, 0, 0), cv2.FILLED)
    # Hands Portion
    hand_results = hands.process(frame_rgb)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        #Highlight index fingertip

        index_tip = hand_landmarks.landmark[8]
        h, w, _ = frame.shape
        ix, iy = int(index_tip.x * w), int(index_tip.y * h)

        #Highlight middle fingertip
        middle_tip = hand_landmarks.landmark[12]
        mx, my = int(middle_tip.x * w), int(middle_tip.y * h)
        
        cv2.circle(frame, (ix, iy), 8, (0, 255, 0), cv2.FILLED)
        cv2.circle(frame, (mx, my), 8, (0, 255, 0), cv2.FILLED)

    cv2.imshow("Hands + Pose Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
