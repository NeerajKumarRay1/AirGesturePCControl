import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import screen_brightness_control as sbc
import time
from pynput.mouse import Controller, Button
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from utils import (apply_exponential_smoothing, is_two_fingers_up, is_pinch, is_middle_thumb_tap, 
                   is_fingers_apart, is_five_fingers_up, is_five_fingers_down, is_pinky_finger_up, 
                   is_pinky_finger_down, is_index_middle_fingers_together, is_full_palm_open, 
                   is_index_finger_up, is_namaste_gesture)

mouse = Controller()
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

cursor_smoothing = 0.8
smoothed_cursor_x, smoothed_cursor_y = None, None
cursor_active, click_ready = True, False
swipe_start_time, last_palm_open_time = None, 0
palm_open_cooldown = 2  # Cooldown for certain gestures

# Initialize volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

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
                    cursor_active, click_ready = False, True
                else:
                    cursor_active, click_ready = True, False

                if cursor_active and is_two_fingers_up(hand_landmarks):
                    screen_width, screen_height = pyautogui.size()
                    cursor_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * screen_width)
                    cursor_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * screen_height)
                    smoothed_cursor_x = apply_exponential_smoothing(cursor_x, smoothed_cursor_x, cursor_smoothing)
                    smoothed_cursor_y = apply_exponential_smoothing(cursor_y, smoothed_cursor_y, cursor_smoothing)
                    mouse.position = (smoothed_cursor_x, smoothed_cursor_y)

                if not cursor_active and click_ready:
                    if is_pinch(hand_landmarks):
                        mouse.click(Button.left, 1)
                        click_ready = False
                    elif is_middle_thumb_tap(hand_landmarks):
                        mouse.click(Button.right, 1)
                        click_ready = False

                if is_pinky_finger_up(hand_landmarks):
                    sbc.set_brightness(min(sbc.get_brightness(display=0)[0] + 10, 100), display=0)
                elif is_pinky_finger_down(hand_landmarks):
                    sbc.set_brightness(max(sbc.get_brightness(display=0)[0] - 10, 0), display=0)

            elif handedness.classification[0].label == "Left":
                if is_index_middle_fingers_together(hand_landmarks):
                    mouse.scroll(0, 0.5 if is_two_fingers_up(hand_landmarks) else -0.5)

                if is_full_palm_open(hand_landmarks) and (current_time - last_palm_open_time > palm_open_cooldown):
                    pyautogui.hotkey('win', 'tab')
                    last_palm_open_time = current_time

                if is_index_finger_up(hand_landmarks):
                    volume.SetMasterVolumeLevelScalar(min(volume.GetMasterVolumeLevelScalar() + 0.05, 1.0), None)
                else:
                    volume.SetMasterVolumeLevelScalar(max(volume.GetMasterVolumeLevelScalar() - 0.05, 0.0), None)

        if len(hand_landmarks_list) == 2:
            hand_landmarks1, handedness1 = hand_landmarks_list[0]
            hand_landmarks2, handedness2 = hand_landmarks_list[1]
            if handedness1.classification[0].label != handedness2.classification[0].label and is_namaste_gesture(hand_landmarks1, hand_landmarks2):
                break

    resized_frame = cv2.resize(frame, (1080, 720))
    cv2.imshow('Hand Gesture', resized_frame)
    cv2.resizeWindow('Hand Gesture', 1080, 720)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
