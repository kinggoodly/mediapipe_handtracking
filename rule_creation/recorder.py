from .features import extract_features

class Recorder:
    def __init__(self):
        self.frames = []
        self.recording = False

    def start(self):
        self.frames = []
        self.recording = True

    def stop(self):
        self.recording = False
        return self.frames

    def update(self, landmarks_list):
        if not self.recording:
            return

    
        if len(landmarks_list) >= 2:
            frame = []
            for lm in landmarks_list:
                frame.append(extract_features(lm))
            self.frames.append(frame)

