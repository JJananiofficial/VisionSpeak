import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import pyttsx3
import threading

# --- Safe Text-to-Speech Function ---
def speak_sentence(text):
    """ Wakes up speaker safely and reads the whole sentence """
    try:
        # We initialize it fresh every time you press spacebar to prevent crashes!
        engine = pyttsx3.init()
        engine.setProperty('rate', 130)
        
        # The comma (", ") is an invisible pause that wakes up the speaker 
        # so the very first word is spoken perfectly!
        engine.say(", " + text)
        engine.runAndWait()
    except Exception as e:
        print(f"Audio Error: {e}")

# --- Load Model & Encoder ---
model = tf.keras.models.load_model("vision_speak_model.h5")
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# --- Initialize MediaPipe (For 2 Hands) ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# --- Sentence Building Variables ---
last_word = ""
word_stable_frames = 0
STABILITY_THRESHOLD = 15 
sentence = [] 
last_added_word = "" 

print("\n" + "="*30)
print("       CONTROLS")
print("="*30)
print("1. Build your sentence with gestures.")
print("2. Click on the VIDEO WINDOW.")
print("3. Press SPACEBAR to Speak the whole sentence.")
print("4. Press 'c' to Clear sentence.")
print("5. Press 'q' to Quit.")
print("="*30 + "\n")

# --- Main Application Loop ---
while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)
    current_word = ""
    confidence = 0.0
    row_data = [0.0] * 84 

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if i >= 2: break
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            wrist_x = hand_landmarks.landmark[0].x
            wrist_y = hand_landmarks.landmark[0].y
            
            for j, lm in enumerate(hand_landmarks.landmark):
                row_data[(i * 42) + (j * 2)] = lm.x - wrist_x
                row_data[(i * 42) + (j * 2) + 1] = lm.y - wrist_y
        
        prediction = model.predict(np.array([row_data]), verbose=0)[0]
        max_idx = np.argmax(prediction)
        confidence = prediction[max_idx]
        
        if confidence > 0.70:
            current_word = encoder.inverse_transform([max_idx])[0]
            
            cv2.putText(frame, f"Detecting: {current_word} ({confidence*100:.1f}%)", 
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
            if current_word == last_word:
                word_stable_frames += 1
            else:
                word_stable_frames = 0
                last_word = current_word
                
            if word_stable_frames > STABILITY_THRESHOLD:
                if current_word != last_added_word: 
                    sentence.append(current_word)
                    last_added_word = current_word
                word_stable_frames = -20 

    # --- UI FIX: Keep text from going off the screen ---
    display_text = " ".join(sentence[-6:]) 
    
    cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, f"Sentence: {display_text}", (10, h - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("VisionSpeak - Sentence Mode", frame)
    
    # --- Keyboard Controls ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        break
    elif key == ord(' ') or key == 13: # SPACEBAR or ENTER
        if len(sentence) > 0:
            full_sentence = " ".join(sentence)
            print(f"\n--> SPEAKING ALOUD: '{full_sentence}'")
            
            # Send the audio to the safe function (uses threading so the video doesn't freeze!)
            threading.Thread(target=speak_sentence, args=(full_sentence,)).start()
            
            # Clear sentence after speaking
            sentence = [] 
            last_added_word = ""
            
    elif key == ord('c'): 
        sentence = []
        last_added_word = ""
        print("\nSentence cleared.")

cap.release()
cv2.destroyAllWindows()