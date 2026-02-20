from __future__ import annotations

import cv2
import numpy as np


def feather_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return mask
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)


def temporal_smooth_frame(prev_frame: np.ndarray, current_frame: np.ndarray, alpha: float) -> np.ndarray:
    alpha = max(0.0, min(1.0, alpha))
    return cv2.addWeighted(prev_frame, alpha, current_frame, 1.0 - alpha, 0.0)


def seamless_blend(
    source: np.ndarray,
    destination: np.ndarray,
    mask: np.ndarray,
    center: tuple[int, int],
) -> np.ndarray:
    return cv2.seamlessClone(source, destination, mask, center, cv2.NORMAL_CLONE)
