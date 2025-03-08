import numpy as np

def apply_exponential_smoothing(new_value, smoothed_value, smoothing_factor):
    if smoothed_value is None:
        return new_value
    return smoothing_factor * new_value + (1 - smoothing_factor) * smoothed_value

def calculate_distance(p1, p2):
    return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

def is_two_fingers_up(hand_landmarks):
    return hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y and hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y

def is_pinch(hand_landmarks):
    return calculate_distance(hand_landmarks.landmark[8], hand_landmarks.landmark[4]) < 0.05

def is_middle_thumb_tap(hand_landmarks):
    return calculate_distance(hand_landmarks.landmark[12], hand_landmarks.landmark[4]) < 0.05

def is_fingers_apart(hand_landmarks):
    return calculate_distance(hand_landmarks.landmark[8], hand_landmarks.landmark[12]) > 0.07

def is_five_fingers_up(hand_landmarks):
    return all(hand_landmarks.landmark[finger].y < hand_landmarks.landmark[finger - 2].y for finger in [8, 12, 16, 20])

def is_five_fingers_down(hand_landmarks):
    return all(hand_landmarks.landmark[finger].y > hand_landmarks.landmark[finger - 2].y for finger in [8, 12, 16, 20])

def is_pinky_finger_up(hand_landmarks):
    return hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y

def is_pinky_finger_down(hand_landmarks):
    return hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y

def is_index_middle_fingers_together(hand_landmarks):
    return calculate_distance(hand_landmarks.landmark[8], hand_landmarks.landmark[12]) < 0.05

def is_full_palm_open(hand_landmarks):
    return is_five_fingers_up(hand_landmarks)

def is_index_finger_up(hand_landmarks):
    return hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y

def is_namaste_gesture(hand_landmarks1, hand_landmarks2):
    return calculate_distance(hand_landmarks1.landmark[0], hand_landmarks2.landmark[0]) < 0.1
