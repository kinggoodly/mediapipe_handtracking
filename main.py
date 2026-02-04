import cv2
import mediapipe as mp
import os
from rule_creation.detector import Detector

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

RULE_DIR = "rule_creation/rules"
detectors = []

for file in os.listdir(RULE_DIR):
    if file.endswith(".json"):
        detectors.append(
            Detector(os.path.join(RULE_DIR, file))
        )

print(f"[INFO] Loaded {len(detectors)} gesture rules")

POPUP_IMAGES = {
    "Gojo": "img/gojo.png",
    "Sukuna": "img/saku.jpg",
    "Mahoraga": "img/mahoraga.png",
    # "Gesture4": "img/Gesture4.png",
    # "Gesture5": "img/Gesture5.png",
    # ...
}

popup_cache = {}
popup_visible = None

for gesture, path in POPUP_IMAGES.items():
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Popup image not found: {path}")
    popup_cache[gesture] = cv2.resize(img, (1280, 1280))

cap = cv2.VideoCapture(0)

print("[INFO] Press ESC to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    landmarks_list = []
    detected_name = None

    if result.multi_hand_landmarks:
        for hand_lm in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_lm,
                mp_hands.HAND_CONNECTIONS
            )
            landmarks_list.append(hand_lm.landmark)

        for detector in detectors:
            if detector.update(landmarks_list):
                detected_name = detector.rule["gesture"]
                break

    if detected_name:
        cv2.putText(
            frame,
            f"Detected: {detected_name}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

    if detected_name in popup_cache:
        if popup_visible != detected_name:
            if popup_visible is not None:
                cv2.destroyWindow(popup_visible)

            cv2.imshow(detected_name, popup_cache[detected_name])
            print(detected_name)
            popup_visible = detected_name
    else:
        if popup_visible is not None:
            cv2.destroyWindow(popup_visible)
            popup_visible = None

    cv2.imshow("Gesture Detect (2 Hands)", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
