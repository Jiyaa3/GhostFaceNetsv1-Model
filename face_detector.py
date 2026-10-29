"""
face_detector.py - YuNet face detection and alignment wrapper for Windows/Linux.
Uses OpenCV cv2.FaceDetectorYN.
"""

import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
import cv2
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass
import numpy as np
from face_align import align_face_112

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_YUNET_PATH = os.path.join(SCRIPT_DIR, "models", "face_detection_yunet.onnx")
FALLBACK_YUNET_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "models", "face_detection_yunet.onnx")


class YuNetDetector:
    def __init__(self, model_path=None, score_threshold=0.6, nms_threshold=0.3):
        if model_path is None or not os.path.exists(model_path):
            if os.path.exists(DEFAULT_YUNET_PATH):
                model_path = DEFAULT_YUNET_PATH
            elif os.path.exists(FALLBACK_YUNET_PATH):
                model_path = FALLBACK_YUNET_PATH
            else:
                raise FileNotFoundError(
                    f"YuNet model file not found at: {DEFAULT_YUNET_PATH} or {FALLBACK_YUNET_PATH}"
                )

        self.model_path = model_path
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.input_size = (320, 320)
        self.detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            self.input_size,
            self.score_threshold,
            self.nms_threshold,
            5000
        )

    def set_input_size(self, width, height):
        if (width, height) != self.input_size:
            self.input_size = (int(width), int(height))
            self.detector.setInputSize(self.input_size)

    def detect_and_align(self, frame):
        """
        Detect faces in the frame and produce aligned 112x112 crops.
        Optimized for 720p/1080p HD feeds: detects at 640-scale for 80+ FPS speed,
        then warps the face directly from the pristine full-res frame.
        """
        h, w = frame.shape[:2]
        
        # Scale for ultra-fast detector inference if frame is HD
        if w > 640:
            scale_x = w / 640.0
            scale_y = h / 360.0
            det_img = cv2.resize(frame, (640, 360))
            self.set_input_size(640, 360)
            _, faces = self.detector.detect(det_img)
            if faces is None:
                return []

            results = []
            for face in faces:
                # Map back to full-resolution frame
                bx = int(face[0] * scale_x)
                by = int(face[1] * scale_y)
                bw = int(face[2] * scale_x)
                bh = int(face[3] * scale_y)
                score = float(face[14])
                landmarks = np.array([
                    [face[4] * scale_x, face[5] * scale_y],    # left eye
                    [face[6] * scale_x, face[7] * scale_y],    # right eye
                    [face[8] * scale_x, face[9] * scale_y],    # nose tip
                    [face[10] * scale_x, face[11] * scale_y],  # left mouth
                    [face[12] * scale_x, face[13] * scale_y]   # right mouth
                ], dtype=np.float32)

                # Warp from the uncompressed full-res frame for peak facial detail
                aligned_face = align_face_112(frame, landmarks)
                results.append({
                    "box": (bx, by, bw, bh),
                    "score": score,
                    "landmarks": landmarks,
                    "aligned_face": aligned_face
                })
            return results
        else:
            self.set_input_size(w, h)
            _, faces = self.detector.detect(frame)
            if faces is None:
                return []

            results = []
            for face in faces:
                box = (int(face[0]), int(face[1]), int(face[2]), int(face[3]))
                score = float(face[14])
                landmarks = np.array([
                    [face[4], face[5]],    # left eye
                    [face[6], face[7]],    # right eye
                    [face[8], face[9]],    # nose tip
                    [face[10], face[11]],  # left mouth
                    [face[12], face[13]]   # right mouth
                ], dtype=np.float32)

                aligned_face = align_face_112(frame, landmarks)
                results.append({
                    "box": box,
                    "score": score,
                    "landmarks": landmarks,
                    "aligned_face": aligned_face
                })
            return results
