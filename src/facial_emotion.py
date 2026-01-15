import mediapipe as mp
import cv2
import numpy as np
import os
from typing import Dict

# 7-class FER labels (common)
FER_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

mp_face = mp.solutions.face_detection

class FacialEmotion:
    def __init__(self, model_path: str = None, min_detection_confidence=0.5):
        self.face_detector = mp_face.FaceDetection(min_detection_confidence=min_detection_confidence)
        self.model = None
        self.model_path = model_path
        if model_path and os.path.exists(model_path):
            try:
                # Lazy import Keras only if model provided
                from tensorflow.keras.models import load_model
                self.model = load_model(model_path)
                print(f"[facial_emotion] Loaded FER model from {model_path}")
            except Exception as e:
                print(f"[facial_emotion] Failed to load model: {e}. Falling back to placeholder.")

    def process(self, frame_rgb) -> Dict:
        """
        frame_rgb: HxWx3 RGB ndarray
        returns dict: {label, confidence, bbox}
        """
        result = {"label": None, "confidence": None, "bbox": None}
        # MediaPipe expects RGB
        image = frame_rgb.copy()
        h, w, _ = image.shape
        detections = self.face_detector.process(image)
        if detections and detections.detections:
            det = detections.detections[0]
            bboxC = det.location_data.relative_bounding_box
            x1 = max(0, int(bboxC.xmin * w))
            y1 = max(0, int(bboxC.ymin * h))
            bw = int(bboxC.width * w)
            bh = int(bboxC.height * h)
            x2 = min(w, x1 + bw)
            y2 = min(h, y1 + bh)
            result["bbox"] = [x1, y1, x2, y2]
            face_img = image[y1:y2, x1:x2]
            if face_img.size == 0:
                return {"label": "neutral", "confidence": 1.0, "bbox": result["bbox"]}
            # Preprocess for mini_XCEPTION style model: 48x48 grayscale
            gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, (48, 48))
            input_arr = resized.astype("float32") / 255.0
            input_arr = np.expand_dims(input_arr, axis=(0, -1))  # shape (1,48,48,1)
            if self.model is not None:
                try:
                    preds = self.model.predict(input_arr)
                    idx = int(np.argmax(preds))
                    result["label"] = FER_LABELS[idx] if idx < len(FER_LABELS) else str(idx)
                    result["confidence"] = float(np.max(preds))
                except Exception as e:
                    result["label"] = "neutral"
                    result["confidence"] = 1.0
            else:
                # Fallback placeholder: return neutral with 1.0
                result["label"] = "neutral"
                result["confidence"] = 1.0
        else:
            result["label"] = None
            result["confidence"] = None
            result["bbox"] = None
        return result

    def close(self):
        self.face_detector.close()
