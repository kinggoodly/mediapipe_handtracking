import cv2
import mediapipe as mp
import os

from rule_creation.recorder import Recorder
from rule_creation.rule_generator import generate_rule
from rule_creation.detector import Detector

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

recorder = Recorder()
detector = None
recorded = False

cap = cv2.VideoCapture(0)

print("R = start record | S = stop & save | D = detect | ESC = quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    landmarks_list = []

    if result.multi_hand_landmarks:
        for hand_lm in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_lm,
                mp_hands.HAND_CONNECTIONS
            )
            landmarks_list.append(hand_lm.landmark)

        # ----- RECORD -----
        recorder.update(landmarks_list)

        # ----- DETECT -----
        if detector and detector.update(landmarks_list):
            cv2.putText(
                frame, "DETECTED",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3
            )

    cv2.putText(
        frame,
        "REC" if recorder.recording else "IDLE",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255) if recorder.recording else (255, 255, 255),
        3
    )

    cv2.imshow("Gesture System (2 Hands)", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

    elif key == ord('r'):
        recorder.start()
        print("Recording... (show 2 hands)")

    elif key == ord('s'):
        frames = recorder.stop()
        generate_rule(frames,"Sukuna")
        print("Rule saved: DivineDogs")
        recorded = True

    elif key == ord('d') and recorded:
        detector = Detector("rule_creation/rules/Sukuna.json")
        print("Detect mode ON")

cap.release()
cv2.destroyAllWindows()
