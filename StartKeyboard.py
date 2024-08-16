##this is tasted on kali linux 2024.2
import cv2
import mediapipe as mp
import numpy as np
import time
import math
from pynput.keyboard import Controller

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Initialize the Keyboard Controller
keyboard = Controller()

# Define the keys on the virtual keyboard
keys = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
    ["Z", "X", "C", "V", "B", "N", "M"]
]

# Keyboard dimensions
key_width = 60
key_height = 60
start_x = 50
start_y = 50

# Variables to track key press
last_pressed_key = None
highlighted_key = None
press_time = 0
press_delay = 0.7  # Seconds to wait between key presses

# Function to draw the virtual keyboard
def draw_keyboard(img, highlighted_key=None):
    for i, row in enumerate(keys):
        for j, key in enumerate(row):
            x = start_x + j * key_width
            y = start_y + i * key_height
            color = (0, 255, 0) if key == highlighted_key else (255, 0, 0)
            cv2.rectangle(img, (x, y), (x + key_width, y + key_height), color, 2)
            cv2.putText(img, key, (x + 20, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Function to detect if a finger is pressing a key
def detect_key_press(x, y):
    for i, row in enumerate(keys):
        for j, key in enumerate(row):
            key_x = start_x + j * key_width
            key_y = start_y + i * key_height
            if key_x < x < key_x + key_width and key_y < y < key_y + key_height:
                return key
    return None

# Function to detect if the first two fingers are touching
def are_fingers_touching(hand_landmarks, img):
    # Get coordinates of the index and middle fingertip (landmark 8 and 12)
    index_finger_tip = hand_landmarks.landmark[8]
    middle_finger_tip = hand_landmarks.landmark[12]
    
    h, w, _ = img.shape
    index_finger_coords = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))
    middle_finger_coords = (int(middle_finger_tip.x * w), int(middle_finger_tip.y * h))
    
    # Calculate distance between the fingertips
    distance = math.hypot(index_finger_coords[0] - middle_finger_coords[0],
                          index_finger_coords[1] - middle_finger_coords[1])
    
    # If the distance is below a certain threshold, the fingers are touching
    if distance < 40:  # Adjusted threshold
        print("Fingers touched!")  # Debugging line
        return True
    return False

# Start capturing video
cap = cv2.VideoCapture(1)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip the image horizontally
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process the image and find hands
    results = hands.process(img_rgb)
    
    # Draw the virtual keyboard
    draw_keyboard(img, highlighted_key)
    
    # Reset highlighted key
    highlighted_key = None
    
    # Check if any hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get coordinates of the index fingertip (landmark 8)
            index_finger_tip = hand_landmarks.landmark[8]
            h, w, _ = img.shape
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            
            # Detect key under the fingertip and highlight it
            highlighted_key = detect_key_press(cx, cy)
            
            # Detect if the first two fingers are touching
            if are_fingers_touching(hand_landmarks, img):
                current_time = time.time()
                if highlighted_key and (last_pressed_key != highlighted_key or current_time - press_time > press_delay):
                    last_pressed_key = highlighted_key
                    press_time = current_time
                    print(f"Key pressed: {highlighted_key}")  # Debugging line
                    keyboard.press(highlighted_key.lower())
                    keyboard.release(highlighted_key.lower())
    
    # Display the image
    cv2.imshow("Virtual Keyboard", img)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
