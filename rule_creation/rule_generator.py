import json
import numpy as np
import os

def generate_rule(frames, gesture_name):
    rule = {}
    hands = len(frames[0])  # 1 หรือ 2 มือ

    for hand_id in range(hands):
        for key in frames[0][hand_id].keys():
            values = [f[hand_id][key] for f in frames]
            mean = np.mean(values)
            std = np.std(values)
            rule[f"h{hand_id}_{key}"] = {
                "min": float(mean - 2*std),
                "max": float(mean + 2*std)
            }

    data = {
        "gesture": gesture_name,
        "hands": hands,
        "allow_partial": True,
        "features": rule,
        "window": 12,
        "vote": 8
    }

    os.makedirs("rule_creation/rules", exist_ok=True)
    with open(f"rule_creation/rules/{gesture_name}.json", "w") as f:
        json.dump(data, f, indent=2)

    return data
