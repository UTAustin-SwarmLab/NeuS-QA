from typing import List
import numpy as np
import cv2


class Mp4Reader():
    def __init__(self, path: str, sample_rate: float = 1.0):
        self.path = path
        self.sample_rate = float(sample_rate)

    def _sampled_frame_indices(self, fps: float, frame_count: int) -> List[int]:
        if fps <= 0:
            fps = 1.0

        duration_sec = frame_count / fps if frame_count > 0 else 0.0
        step_sec = 1.0 / self.sample_rate

        times = [t for t in np.arange(0.0, duration_sec + 1e-9, step_sec)]
        idxs = sorted(set(int(round(t * fps)) for t in times if t * fps < frame_count))
        if not idxs and frame_count > 0:
            idxs = [0]
        return idxs

    def read_video(self):
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        frame_idxs = self._sampled_frame_indices(fps, frame_count)

        images: List[np.ndarray] = []
        for idx in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            images.append(frame_rgb)

        if (width == 0 or height == 0) and images:
            height, width = images[0].shape[:2]

        video_info = {
            "frame_width": width,
            "frame_height": height,
            "frame_count": frame_count,
            "fps": float(fps) if fps else None,
        }

        cap.release()
        output = {
            "video_path": self.path,
            "sample_rate": self.sample_rate,
            "video_info": video_info,
            "images": images,
        }
        return output

