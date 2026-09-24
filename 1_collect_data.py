import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# Initialize MediaPipe Hands (Changed to max_num_hands=2)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Set up the l
# label you are recording
label = "Sorry"  # Change this for each gesture you record
num_samples = 200 

csv_file = "gesture_dataset.csv"

# Create CSV header for TWO hands (84 coordinates)
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ["label"] + [f"coord_{i}" for i in range(84)] 
        writer.writerow(header)

cap = cv2.VideoCapture(0)
count = 0

print(f"Press 's' to start collecting data for gesture: {label}")
start_collection = False

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1) 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)
    
    # Create an empty array of 84 zeros. (If only 1 hand is shown, the rest stay zero)
    row_data = [0.0] * 84 
    
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if i >= 2: break # Make sure we only process up to 2 hands
            
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Pre-processing: Normalize relative to wrist
            wrist_x = hand_landmarks.landmark[0].x
            wrist_y = hand_landmarks.landmark[0].y
            
            # Fill the row_data array. 
            # i=0 fills index 0-41 (Hand 1), i=1 fills index 42-83 (Hand 2)
            for j, lm in enumerate(hand_landmarks.landmark):
                row_data[(i * 42) + (j * 2)] = lm.x - wrist_x
                row_data[(i * 42) + (j * 2) + 1] = lm.y - wrist_y
                
        if start_collection and count < num_samples:
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([label] + row_data)
            
            count += 1
            cv2.putText(frame, f"Collected: {count}/{num_samples}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
    cv2.imshow("Data Collection (2 Hands)", frame)
    key = cv2.waitKey(1)
    if key == ord('s'): start_collection = True
    if key == ord('q') or count >= num_samples: break

cap.release()
cv2.destroyAllWindows()
print(f"Successfully collected {count} samples for '{label}'.")