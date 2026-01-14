import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic

# A simple wrapper that maintains a Holistic object
class PoseGestureEstimator:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_rgb):
        """
        frame_rgb: HxWx3 RGB numpy array (uint8)
        returns dict with pose and hands landmarks (if available)
        """
        res = self.holistic.process(frame_rgb)
        out = {"pose": None, "left_hand": None, "right_hand": None}
        if res.pose_landmarks:
            out["pose"] = [(lm.x, lm.y, lm.z, lm.visibility) for lm in res.pose_landmarks.landmark]
        if res.left_hand_landmarks:
            out["left_hand"] = [(lm.x, lm.y, lm.z) for lm in res.left_hand_landmarks.landmark]
        if res.right_hand_landmarks:
            out["right_hand"] = [(lm.x, lm.y, lm.z) for lm in res.right_hand_landmarks.landmark]
        return out

    def close(self):
        self.holistic.close()
