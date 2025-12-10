"""
Lightweight video recorder for inference runs.
Captures grayscale frames and writes them to outputs/ as an mp4.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional

import mediapy as media
import numpy as np

from config import OUTPUTS_DIR


def _default_output_path(prefix: str = "inference") -> Path:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUTS_DIR / f"{prefix}_{timestamp}.mp4"


class VideoMonitor:
    def __init__(self, path: Optional[Path | str] = None, fps: int = 60):
        self.output_path = Path(path) if path else _default_output_path()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.writer: Optional[media.VideoWriter] = None

    def write(self, frame) -> None:
        """Accept a frame shaped (H,W) or (H,W,1) and append to the video."""
        arr = np.asarray(frame)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]

        if self.writer is None:
            self.writer = media.VideoWriter(
                self.output_path, arr.shape, fps=self.fps, input_format="gray"
            )
            self.writer.__enter__()

        self.writer.add_image(arr)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
