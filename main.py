import cv2
from pynput.mouse import Controller, Button
import mediapipe as mp
import numpy as np
import pyautogui
import screen_brightness_control as sbc
import time  # Add this import for time tracking
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import math

mouse = Controller()

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

cursor_smoothing = 0.8  # Increased smoothing factor for better stability
smoothed_cursor_x = None
smoothed_cursor_y = None
cursor_active = True  # Flag to enable/disable cursor movement
click_ready = False  # Flag to indicate readiness for clicking
swipe_start_time = None  # Add this variable to track swipe start time
last_palm_open_time = 0  # Add this variable to track the last time the palm was open
palm_open_cooldown = 2  # Cooldown period in seconds

# Initialize volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

def apply_exponential_smoothing(new_value, smoothed_value, smoothing_factor):
    if smoothed_value is None:
        return new_value
    return smoothing_factor * new_value + (1 - smoothing_factor) * smoothed_value

def is_two_fingers_up(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
    return index_tip < index_pip and middle_tip < middle_pip and ring_tip > middle_tip and pinky_tip > middle_tip

def is_pinch(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    distance = np.linalg.norm(np.array([index_tip.x, index_tip.y, index_tip.z]) - np.array([thumb_tip.x, thumb_tip.y, thumb_tip.z]))
    return distance < 0.05

def is_middle_thumb_tap(hand_landmarks):
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    distance = np.linalg.norm(np.array([middle_tip.x, middle_tip.y, middle_tip.z]) - np.array([thumb_tip.x, thumb_tip.y, thumb_tip.z]))
    return distance < 0.05

def is_fingers_apart(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    distance = np.linalg.norm(np.array([index_tip.x, index_tip.y, index_tip.z]) - np.array([middle_tip.x, middle_tip.y, middle_tip.z]))
    return distance > 0.07

def is_five_fingers_up(hand_landmarks):
    return all(hand_landmarks.landmark[finger].y < hand_landmarks.landmark[finger - 2].y for finger in 
               [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 
                mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP])

def is_five_fingers_down(hand_landmarks):
    return all(hand_landmarks.landmark[finger].y > hand_landmarks.landmark[finger - 2].y for finger in 
               [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 
                mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP])

def is_pinky_finger_up(hand_landmarks):
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y
    return pinky_tip < pinky_pip

def is_pinky_finger_down(hand_landmarks):
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
    return pinky_tip > pinky_pip and index_tip > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and \
           middle_tip > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and \
           ring_tip > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y and \
           thumb_tip > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y

def is_index_middle_fingers_together(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    distance = np.linalg.norm(np.array([index_tip.x, index_tip.y, index_tip.z]) - np.array([middle_tip.x, middle_tip.y, middle_tip.z]))
    return distance < 0.05

def is_full_palm_open(hand_landmarks):
    return is_five_fingers_up(hand_landmarks)

def is_index_finger_up(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    return index_tip < index_pip

def is_namaste_gesture(hand_landmarks1, hand_landmarks2):
    wrist1 = hand_landmarks1.landmark[mp_hands.HandLandmark.WRIST]
    wrist2 = hand_landmarks2.landmark[mp_hands.HandLandmark.WRIST]
    distance = np.linalg.norm(np.array([wrist1.x, wrist1.y]) - np.array([wrist2.x, wrist2.y]))
    return distance < 0.1

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    frame_width = frame.shape[1]
    current_time = time.time()
    if results.multi_hand_landmarks:
        hand_landmarks_list = list(zip(results.multi_hand_landmarks, results.multi_handedness))
        for hand_landmarks, handedness in hand_landmarks_list:
            if handedness.classification[0].label == "Right":
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                if is_fingers_apart(hand_landmarks):
                    cursor_active = False
                    click_ready = True
                else:
                    cursor_active = True
                    click_ready = False
                if cursor_active and is_two_fingers_up(hand_landmarks):
                    screen_width, screen_height = pyautogui.size()
                    cursor_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * screen_width)
                    cursor_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * screen_height)
                    smoothed_cursor_x = apply_exponential_smoothing(cursor_x, smoothed_cursor_x, cursor_smoothing)
                    smoothed_cursor_y = apply_exponential_smoothing(cursor_y, smoothed_cursor_y, cursor_smoothing)
                    mouse.position = (smoothed_cursor_x, smoothed_cursor_y)
                if not cursor_active and click_ready and is_pinch(hand_landmarks):
                    mouse.click(Button.left, 1)
                    click_ready = False
                if not cursor_active and click_ready and is_middle_thumb_tap(hand_landmarks):
                    mouse.click(Button.right, 1)
                    click_ready = False
                if is_pinky_finger_up(hand_landmarks):
                    #print("Pinky finger up detected")  # Debug print
                    sbc.set_brightness(min(sbc.get_brightness(display=0)[0] + 10, 100), display=0)
                elif is_pinky_finger_down(hand_landmarks):
                    #print("Pinky finger down detected")  # Debug print
                    sbc.set_brightness(max(sbc.get_brightness(display=0)[0] - 10, 0), display=0)
            elif handedness.classification[0].label == "Left":
                if is_index_middle_fingers_together(hand_landmarks):
                    if is_two_fingers_up(hand_landmarks):
                        mouse.scroll(0, 0.5)  # Scroll up
                    elif is_five_fingers_down(hand_landmarks):
                        mouse.scroll(0, -0.5)  # Scroll down
                if is_full_palm_open(hand_landmarks) and (current_time - last_palm_open_time > palm_open_cooldown):
                    pyautogui.hotkey('win', 'tab')
                    last_palm_open_time = current_time  # Update the last palm open time
                if is_index_finger_up(hand_landmarks):
                    volume.SetMasterVolumeLevelScalar(min(volume.GetMasterVolumeLevelScalar() + 0.05, 1.0), None)
                else:
                    volume.SetMasterVolumeLevelScalar(max(volume.GetMasterVolumeLevelScalar() - 0.05, 0.0), None)
        if len(hand_landmarks_list) == 2:
            hand_landmarks1, handedness1 = hand_landmarks_list[0]
            hand_landmarks2, handedness2 = hand_landmarks_list[1]
            if handedness1.classification[0].label != handedness2.classification[0].label:
                if is_namaste_gesture(hand_landmarks1, hand_landmarks2):
                    break
    resized_frame = cv2.resize(frame, (1080, 720))
    cv2.imshow('Hand Gesture', resized_frame)
    cv2.resizeWindow('Hand Gesture', 1080, 720)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
