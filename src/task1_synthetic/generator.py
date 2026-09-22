"""
Synthetic Industrial Image Generator
Simulates machined metal component surfaces and injects realistic defects
with parameterized severity levels and ground-truth metadata.
"""

import os
import json
import math
import random
from enum import Enum
from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np


class DefectType(str, Enum):
    NORMAL = "normal"
    SCRATCH = "scratch"
    CRACK = "crack"
    CORROSION = "corrosion"
    DIMENSIONAL = "dimensional"


class SyntheticIndustrialGenerator:
    """Generates realistic synthetic industrial metal surfaces with defect injections."""

    def __init__(self, img_size: Tuple[int, int] = (512, 512), seed: Optional[int] = None):
        self.img_size = img_size
        self.height, self.width = img_size
        if seed is not None:
            self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        """Sets random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)

    def generate_base_surface(self, base_gray: int = 175) -> np.ndarray:
        """
        Generates a realistic brushed metal component surface:
        Directional brushed grain lines, microscopic surface texture,
        and subtle illumination gradient.
        """
        # 1. Base uniform metallic background
        surface = np.full((self.height, self.width), base_gray, dtype=np.float32)

        # 2. Add directional brushed grain (horizontal micro-streaks)
        grain_noise = np.random.normal(0, 7.0, (self.height, 1)).astype(np.float32)
        grain_streaks = np.repeat(grain_noise, self.width, axis=1)
        surface += grain_streaks

        # 3. Add isotropic fine texture noise
        fine_noise = np.random.normal(0, 3.5, (self.height, self.width)).astype(np.float32)
        surface += fine_noise

        # 4. Add subtle radial lighting gradient (factory illumination)
        y, x = np.ogrid[:self.height, :self.width]
        center_y, center_x = self.height / 2.0, self.width / 2.0
        max_dist = math.sqrt(center_x**2 + center_y**2)
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        vignette = 1.0 - 0.12 * (dist_from_center / max_dist)
        surface *= vignette

        # Clip and convert to uint8 3-channel BGR image
        surface_uint8 = np.clip(surface, 0, 255).astype(np.uint8)
        bgr_image = cv2.cvtColor(surface_uint8, cv2.COLOR_GRAY2BGR)
        return bgr_image

    def _inject_scratch(self, image: np.ndarray, severity: float) -> Tuple[np.ndarray, np.ndarray]:
        """Injects a linear/curved surface scratch scaled by severity."""
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        img_out = image.copy()

        scale = min(self.width, self.height) / 512.0
        base_len = int((40 + severity * 28) * scale)
        thickness = max(1, int((1 + severity * 0.4) * scale))
        darkness = int(50 + severity * 8)

        margin = max(10, int(min(self.width, self.height) * 0.12))
        x0 = random.randint(margin, max(margin + 1, self.width - margin))
        y0 = random.randint(margin, max(margin + 1, self.height - margin))
        angle = random.uniform(0, 2 * math.pi)

        points = [(x0, y0)]
        num_segments = random.randint(4, 7)
        cur_x, cur_y = x0, y0
        seg_len = max(3.0, base_len / num_segments)

        for _ in range(num_segments):
            angle += random.uniform(-0.35, 0.35)
            cur_x = int(cur_x + seg_len * math.cos(angle))
            cur_y = int(cur_y + seg_len * math.sin(angle))
            cur_x = max(5, min(self.width - 5, cur_x))
            cur_y = max(5, min(self.height - 5, cur_y))
            points.append((cur_x, cur_y))

        pts_array = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(mask, [pts_array], isClosed=False, color=255, thickness=thickness)

        scratch_region = mask > 0
        img_out[scratch_region] = np.clip(
            img_out[scratch_region].astype(np.int16) - darkness, 0, 255
        ).astype(np.uint8)

        highlight_mask = cv2.dilate(mask, np.ones((2, 2), np.uint8)) - mask
        hl_region = highlight_mask > 0
        img_out[hl_region] = np.clip(
            img_out[hl_region].astype(np.int16) + int(darkness * 0.4), 0, 255
        ).astype(np.uint8)

        return img_out, mask

    def _inject_crack(self, image: np.ndarray, severity: float) -> Tuple[np.ndarray, np.ndarray]:
        """Injects a jagged, branching structural crack."""
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        img_out = image.copy()

        scale = min(self.width, self.height) / 512.0
        margin = max(10, int(min(self.width, self.height) * 0.15))
        start_x = random.randint(margin, max(margin + 1, self.width - margin))
        start_y = random.randint(margin, max(margin + 1, self.height - margin))

        num_steps = max(15, int((40 + severity * 20) * scale))
        step_len = max(2.0, 3.0 * scale)
        thickness = max(1, int((1 + severity * 0.35) * scale))
        base_angle = random.uniform(0, 2 * math.pi)

        branches = [([(start_x, start_y)], base_angle, num_steps)]
        all_segments = []

        while branches:
            path, cur_angle, steps_left = branches.pop(0)
            cx, cy = path[-1]
            for _ in range(steps_left):
                cur_angle += random.uniform(-0.6, 0.6)
                cx += step_len * math.cos(cur_angle)
                cy += step_len * math.sin(cur_angle)
                cx = max(5, min(self.width - 5, cx))
                cy = max(5, min(self.height - 5, cy))
                path.append((int(cx), int(cy)))

                if random.random() < (0.02 + severity * 0.008) and len(branches) < 3:
                    branch_angle = cur_angle + random.choice([-1.0, 1.0]) * random.uniform(0.5, 1.2)
                    branches.append(([(int(cx), int(cy))], branch_angle, int(steps_left * 0.6)))

            all_segments.append(path)

        for path in all_segments:
            if len(path) > 1:
                pts_arr = np.array(path, np.int32).reshape((-1, 1, 2))
                cv2.polylines(mask, [pts_arr], isClosed=False, color=255, thickness=thickness)

        crack_region = mask > 0
        img_out[crack_region] = np.clip(
            img_out[crack_region].astype(np.int16) - 130, 0, 255
        ).astype(np.uint8)

        return img_out, mask

    def _inject_corrosion(self, image: np.ndarray, severity: float) -> Tuple[np.ndarray, np.ndarray]:
        """Injects clustered pitting and oxidation patches."""
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        img_out = image.copy()

        scale = min(self.width, self.height) / 512.0
        num_clusters = max(1, int(severity * 0.9))
        margin = max(10, int(min(self.width, self.height) * 0.15))

        for _ in range(num_clusters):
            center_x = random.randint(margin, max(margin + 1, self.width - margin))
            center_y = random.randint(margin, max(margin + 1, self.height - margin))
            cluster_radius = max(5, int((20 + severity * 7) * scale))
            num_pits = max(8, int(25 + severity * 20))

            for _ in range(num_pits):
                offset_r = random.gauss(0, cluster_radius / 2.5)
                offset_theta = random.uniform(0, 2 * math.pi)
                pit_x = int(center_x + offset_r * math.cos(offset_theta))
                pit_y = int(center_y + offset_r * math.sin(offset_theta))

                if 2 <= pit_x < self.width - 2 and 2 <= pit_y < self.height - 2:
                    pit_size = max(1, random.randint(1, max(2, int((2 + severity * 0.8) * scale))))
                    cv2.circle(mask, (pit_x, pit_y), pit_size, 255, -1)

        corrosion_region = mask > 0
        img_out[corrosion_region, 0] = np.clip(img_out[corrosion_region, 0].astype(np.int16) - 70, 0, 255).astype(np.uint8)
        img_out[corrosion_region, 1] = np.clip(img_out[corrosion_region, 1].astype(np.int16) - 55, 0, 255).astype(np.uint8)
        img_out[corrosion_region, 2] = np.clip(img_out[corrosion_region, 2].astype(np.int16) - 20, 0, 255).astype(np.uint8)

        return img_out, mask

    def _inject_dimensional(self, image: np.ndarray, severity: float) -> Tuple[np.ndarray, np.ndarray]:
        """Injects edge deformation, chipped corner, or boundary flaw."""
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        img_out = image.copy()

        scale = min(self.width, self.height) / 512.0
        flaw_size = max(10, int((25 + severity * 12) * scale))
        flaw_type = random.choice(["corner_chip", "edge_notch"])

        if flaw_type == "corner_chip":
            corner = random.choice(["top_left", "top_right", "bottom_left", "bottom_right"])
            if corner == "top_left":
                pts = np.array([[0, 0], [flaw_size, 0], [0, flaw_size]], np.int32)
            elif corner == "top_right":
                pts = np.array([[self.width, 0], [self.width - flaw_size, 0], [self.width, flaw_size]], np.int32)
            elif corner == "bottom_left":
                pts = np.array([[0, self.height], [flaw_size, self.height], [0, self.height - flaw_size]], np.int32)
            else:
                pts = np.array([[self.width, self.height], [self.width - flaw_size, self.height], [self.width, self.height - flaw_size]], np.int32)
            cv2.fillPoly(mask, [pts], 255)
        else:
            side = random.choice(["top", "bottom", "left", "right"])
            radius = max(6, int(flaw_size * 0.7))
            min_bound = max(15, radius + 5)
            if side == "top":
                cx = random.randint(min_bound, max(min_bound + 1, self.width - min_bound))
                cv2.circle(mask, (cx, 0), radius, 255, -1)
            elif side == "bottom":
                cx = random.randint(min_bound, max(min_bound + 1, self.width - min_bound))
                cv2.circle(mask, (cx, self.height), radius, 255, -1)
            elif side == "left":
                cy = random.randint(min_bound, max(min_bound + 1, self.height - min_bound))
                cv2.circle(mask, (0, cy), radius, 255, -1)
            else:
                cy = random.randint(min_bound, max(min_bound + 1, self.height - min_bound))
                cv2.circle(mask, (self.width, cy), radius, 255, -1)

        void_region = mask > 0
        img_out[void_region] = [20, 20, 20]
        return img_out, mask

    def generate_sample(
        self,
        defect_type: DefectType = DefectType.NORMAL,
        severity: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray]:
        """
        Generates a single synthetic component with defect and ground-truth metadata.
        """
        base_img = self.generate_base_surface()

        if defect_type == DefectType.NORMAL:
            severity_score = 0.0
            severity_level = "PASS"
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            img_out = base_img
            bbox = None
            pixel_ratio = 0.0
        else:
            if severity is None:
                severity_score = round(random.uniform(1.0, 10.0), 1)
            else:
                severity_score = round(max(1.0, min(10.0, float(severity))), 1)

            if severity_score <= 4.0:
                severity_level = "LOW"
            elif severity_score <= 7.0:
                severity_level = "MEDIUM"
            else:
                severity_level = "CRITICAL"

            if defect_type == DefectType.SCRATCH:
                img_out, mask = self._inject_scratch(base_img, severity_score)
            elif defect_type == DefectType.CRACK:
                img_out, mask = self._inject_crack(base_img, severity_score)
            elif defect_type == DefectType.CORROSION:
                img_out, mask = self._inject_corrosion(base_img, severity_score)
            elif defect_type == DefectType.DIMENSIONAL:
                img_out, mask = self._inject_dimensional(base_img, severity_score)
            else:
                raise ValueError(f"Unsupported defect type: {defect_type}")

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                all_pts = np.vstack(contours)
                x, y, w, h = cv2.boundingRect(all_pts)
                bbox = [int(x), int(y), int(w), int(h)]
            else:
                bbox = None

            defect_pixels = int(np.count_nonzero(mask))
            pixel_ratio = round(defect_pixels / (self.height * self.width), 5)

        metadata = {
            "defect_type": defect_type.value,
            "severity_score": severity_score,
            "severity_level": severity_level,
            "bounding_box": bbox,
            "affected_pixel_ratio": pixel_ratio,
            "image_dimensions": [self.width, self.height, 3]
        }

        return img_out, metadata, mask

    def generate_dataset(
        self,
        output_dir: str,
        samples_per_class: int = 20,
        seed: Optional[int] = 42
    ) -> Dict[str, Any]:
        """Generates a balanced dataset of synthetic images with ground-truth metadata.json."""
        if seed is not None:
            self.set_seed(seed)

        img_dir = os.path.join(output_dir, "images")
        mask_dir = os.path.join(output_dir, "masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        dataset_metadata = {}
        sample_count = 0

        for dtype in DefectType:
            for i in range(samples_per_class):
                if dtype == DefectType.NORMAL:
                    severity = 0.0
                else:
                    severity = round(1.0 + (i / max(1, samples_per_class - 1)) * 9.0, 1)

                img, meta, mask = self.generate_sample(dtype, severity=severity)
                filename = f"{dtype.value}_{i:03d}.png"
                meta["filename"] = filename

                cv2.imwrite(os.path.join(img_dir, filename), img)
                cv2.imwrite(os.path.join(mask_dir, filename), mask)

                dataset_metadata[filename] = meta
                sample_count += 1

        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(dataset_metadata, f, indent=2)

        summary = {
            "total_samples": sample_count,
            "classes": [d.value for d in DefectType],
            "samples_per_class": samples_per_class,
            "metadata_path": metadata_path,
            "images_dir": img_dir
        }
        return summary