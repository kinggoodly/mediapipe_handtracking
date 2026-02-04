import json
from collections import deque
from .features import extract_features

class Detector:
    def __init__(self, rule_path):
        with open(rule_path) as f:
            self.rule = json.load(f)

        self.hands_required = self.rule["hands"]
        self.buffer = deque(maxlen=self.rule["window"])

    def match(self, landmarks_list):
        required = self.rule["hands"]
        allow_partial = self.rule.get("allow_partial", False)

        # ---------- HAND COUNT LOGIC ----------
        if len(landmarks_list) < required:
            if not allow_partial:
                return False
            # ใช้เฉพาะมือแรกแทน
            landmarks_list = landmarks_list[:1]

        # ---------- FEATURE CHECK ----------
        for i, lm in enumerate(landmarks_list):
            features = extract_features(lm)

            for key, bound in self.rule["features"].items():
                if not key.startswith(f"h{i}_"):
                    continue

                fkey = key.replace(f"h{i}_", "")
                val = features[fkey]

                if not (bound["min"] <= val <= bound["max"]):
                    return False

        return True



    def update(self, landmarks_list):
        ok = self.match(landmarks_list)
        self.buffer.append(ok)
        return self.buffer.count(True) >= self.rule["vote"]
