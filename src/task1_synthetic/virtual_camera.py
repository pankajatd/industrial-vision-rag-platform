"""
Virtual Camera Capture Simulator
Mimics OpenCV cv2.VideoCapture interface to stream synthetic industrial frames
either from disk or dynamically generated on-the-fly.
"""

import os
import time
import json
import random
from typing import Optional, Tuple, Dict, Any, List
import cv2
import numpy as np

from .generator import SyntheticIndustrialGenerator, DefectType


class VirtualCameraCapture:
    """
    Drop-in replacement for cv2.VideoCapture.
    Streams synthetic frames with configurable FPS, looping, and metadata tracking.
    """

    def __init__(
        self,
        source_dir: Optional[str] = None,
        generator: Optional[SyntheticIndustrialGenerator] = None,
        fps: float = 0.0,
        loop: bool = True
    ):
        """
        Args:
            source_dir: Path to directory containing 'images' and 'metadata.json'.
            generator: SyntheticIndustrialGenerator instance for dynamic mode.
            fps: Simulated frame rate (0.0 for maximum throughput without artificial sleep).
            loop: Whether to loop back to the beginning when streaming from folder.
        """
        self.source_dir = source_dir
        self.fps = fps
        self.loop = loop
        self.is_opened = True
        self.frame_idx = 0
        self.last_frame_time = 0.0
        self.current_metadata: Optional[Dict[str, Any]] = None

        if source_dir and os.path.exists(source_dir):
            self.mode = "folder"
            self.img_dir = os.path.join(source_dir, "images")
            metadata_path = os.path.join(source_dir, "metadata.json")

            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    self.metadata_store = json.load(f)
            else:
                self.metadata_store = {}

            if os.path.exists(self.img_dir):
                self.image_files = sorted([
                    f for f in os.listdir(self.img_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
                ])
            else:
                self.image_files = []

            self.generator = None
        else:
            self.mode = "dynamic"
            self.generator = generator or SyntheticIndustrialGenerator()
            self.image_files = []
            self.metadata_store = {}

    def isOpened(self) -> bool:
        """Returns True if the virtual camera is active and ready to stream."""
        if not self.is_opened:
            return False
        if self.mode == "folder":
            return len(self.image_files) > 0
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Grabs and returns the next synthetic video frame, matching cv2.VideoCapture.read().
        """
        if not self.isOpened():
            return False, None

        if self.fps > 0.0:
            target_interval = 1.0 / self.fps
            elapsed = time.time() - self.last_frame_time
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)
            self.last_frame_time = time.time()

        if self.mode == "folder":
            if self.frame_idx >= len(self.image_files):
                if self.loop and len(self.image_files) > 0:
                    self.frame_idx = 0
                else:
                    return False, None

            filename = self.image_files[self.frame_idx]
            img_path = os.path.join(self.img_dir, filename)
            frame = cv2.imread(img_path)

            self.current_metadata = self.metadata_store.get(filename, {
                "filename": filename,
                "defect_type": "unknown",
                "severity_score": 0.0
            })
            self.frame_idx += 1

            if frame is None:
                return False, None
            return True, frame

        else:
            dtype = random.choice(list(DefectType))
            frame, metadata, _ = self.generator.generate_sample(defect_type=dtype)
            metadata["frame_index"] = self.frame_idx
            self.current_metadata = metadata
            self.frame_idx += 1
            return True, frame

    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """Returns ground-truth metadata associated with the most recently read frame."""
        return self.current_metadata

    def release(self) -> None:
        """Closes the virtual camera stream."""
        self.is_opened = False

    def get(self, prop_id: int) -> float:
        """Mimics cv2.VideoCapture.get() for common properties."""
        if prop_id == cv2.CAP_PROP_FPS:
            return float(self.fps)
        elif prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self.image_files)) if self.mode == "folder" else -1.0
        elif prop_id == cv2.CAP_PROP_POS_FRAMES:
            return float(self.frame_idx)
        return 0.0