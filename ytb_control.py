import cv2
import mediapipe as mp
import pyautogui
import time
import speech_recognition as sr

# Initialize camera and Mediapipe
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 15)  # limit FPS for smoothness
hands = mp.solutions.hands.Hands(max_num_hands=1)
draw = mp.solutions.drawing_utils

# Voice commands and their actions
commands = {
    "play": lambda: pyautogui.press("space"),
    "sound": lambda: pyautogui.press("m"),
    "yes": lambda: pyautogui.press("volumeup"),
    "no": lambda: pyautogui.press("volumedown"),
    "next": lambda: pyautogui.press("nexttrack"),
    "previous": lambda: pyautogui.press("prevtrack"),
    "restart": lambda: pyautogui.press("0")
}

recognizer = sr.Recognizer()
mic = sr.Microphone()
prev_action_time = 0

# ✋ Hand detection helper
def fingers_up(hand_landmarks):
    fingers = []
    tip_ids = [4, 8, 12, 16, 20]
    if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)
    for id in range(1, 5):
        if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

# 🎤 Voice recognition background callback
def callback(recognizer, audio):
    try:
        text = recognizer.recognize_google(audio).lower()
        print("[Voice] You said:", text)
        for command in commands:
            if command in text:
                print(f"[Voice] Triggering: {command}")
                commands[command]()
                break
    except sr.UnknownValueError:
        print("[Voice] Could not understand.")
    except sr.RequestError as e:
        print(f"[Voice] API error: {e}")

# Start listening in background (non-blocking)
with mic as source:
    recognizer.adjust_for_ambient_noise(source)
stop_listening = recognizer.listen_in_background(mic, callback)

# 🖐 Main loop
while True:
    success, img = cap.read()
    if not success:
        continue

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            draw.draw_landmarks(img, handLms, mp.solutions.hands.HAND_CONNECTIONS)
            finger_states = fingers_up(handLms)
            total_fingers = sum(finger_states)

            current_time = time.time()
            if total_fingers == 5 and current_time - prev_action_time > 1:
                pyautogui.press('space')
                print("[Gesture] Open palm - Play/Pause")
                prev_action_time = current_time
            elif total_fingers == 1 and current_time - prev_action_time > 1.5:
                pyautogui.press('m')
                print("[Gesture] One finger - Mute/Unmute")
                prev_action_time = current_time
            elif total_fingers == 2 and current_time - prev_action_time > 1.5:
                pyautogui.press('volumeup')
                print("[Gesture] Two fingers - Volume Up")
                prev_action_time = current_time
            elif total_fingers == 3 and current_time - prev_action_time > 1.5:
                pyautogui.press('volumedown')
                print("[Gesture] Three fingers - Volume Down")
                prev_action_time = current_time

    cv2.imshow("Hand + Voice Control", img)
    if cv2.waitKey(1) == ord('q'):
        break

stop_listening()
cap.release()
cv2.destroyAllWindows()
