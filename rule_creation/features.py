import numpy as np
import math

def angle(a, b, c):
    ba = np.array([a.x-b.x, a.y-b.y])
    bc = np.array([c.x-b.x, c.y-b.y])
    cos = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc))
    cos = np.clip(cos, -1.0, 1.0)
    return math.degrees(math.acos(cos))

def dist(a, b):
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)

def extract_features(lm):
    return {
        "thumb_index_dist": dist(lm[4], lm[8]),
        "index_angle": angle(lm[5], lm[6], lm[8]),
        "middle_angle": angle(lm[9], lm[10], lm[12]),
        "ring_angle": angle(lm[13], lm[14], lm[16]),
        "pinky_angle": angle(lm[17], lm[18], lm[20]),
    }
