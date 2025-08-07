"""
Simple video screenshot utility.

Extract screenshots from video files at regular time intervals.
"""

import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional

import cv2
from pathlib import Path

logger = logging.getLogger(__name__)


def _expand_path(path: str) -> Path:
    """
    Expand a path, handling tilde (~) for home directory.
    
    Args:
        path: Path string that may contain tilde
        
    Returns:
        Expanded Path object
    """
    return Path(os.path.expanduser(path))


@dataclass
class ScreenshotInfo:
    """Information about an extracted screenshot."""
    frame_number: int
    timestamp: float  # seconds
    timestamp_str: str  # formatted as HH:MM:SS
    file_path: str


class VideoScreenshotExtractor:
    """
    Extract screenshots from video files at regular intervals.
    
    Simple utility to take screenshots every X seconds from a video file.
    """
    
    def __init__(self, video_path: str):
        """
        Initialize the screenshot extractor.
        
        Args:
            video_path: Path to the video file
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        
        logger.info(
            f"Video loaded: {self.fps:.2f} FPS, "
            f"{self.frame_count} frames, "
            f"{self.duration:.2f}s duration"
        )
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - release video capture."""
        self.release()
    
    def release(self) -> None:
        """Release the video capture object."""
        if self.cap:
            self.cap.release()
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds into HH:MM:SS string."""
        return str(timedelta(seconds=int(seconds)))
    
    def extract_screenshots(
        self,
        interval_seconds: float = 2.0,
        start_time: float = 0.0,
        output_dir: Optional[str] = "~/Downloads",
        filename_prefix: str = "screenshot"
    ) -> List[ScreenshotInfo]:
        """
        Extract screenshots at regular time intervals.
        
        Args:
            interval_seconds: Time interval between screenshots (default: 2.0)
            start_time: Start time in seconds (default: 0.0)
            output_dir: Directory to save screenshots (default: ~/Downloads)
            filename_prefix: Prefix for saved files
            
        Returns:
            List of ScreenshotInfo objects
        """
        if interval_seconds <= 0:
            raise ValueError("Interval must be positive")
        if start_time < 0:
            raise ValueError("Start time must be non-negative")
        
        screenshots = []
        frame_interval = int(self.fps * interval_seconds)
        start_frame = int(start_time * self.fps)
        
        # Handle output directory with tilde expansion
        output_path = _expand_path(output_dir) if output_dir else _expand_path("~/Downloads")
        output_path.mkdir(parents=True, exist_ok=True)
        
        for frame_num in range(start_frame, self.frame_count, frame_interval):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            timestamp = frame_num / self.fps
            timestamp_str = self._format_timestamp(timestamp)
            
            screenshot_info = ScreenshotInfo(
                frame_number=frame_num,
                timestamp=timestamp,
                timestamp_str=timestamp_str,
                file_path=""
            )
            
            # Always save screenshots since we have a default output directory
            # Use timestamp-based filename for chronological sorting
            timestamp_seconds = int(timestamp)
            filename = (f"{filename_prefix}_{timestamp_seconds:06d}s_"
                       f"{timestamp_str.replace(':', '-')}.jpg")
            file_path = output_path / filename
            cv2.imwrite(str(file_path), frame)
            screenshot_info.file_path = str(file_path)
            
            screenshots.append(screenshot_info)
            logger.debug(
                f"Extracted screenshot at {timestamp_str} "
                f"(frame {frame_num})"
            )
        
        logger.info(
            f"Extracted {len(screenshots)} screenshots "
            f"at {interval_seconds}s intervals"
        )
        return screenshots


def extract_screenshots(
    video_path: str,
    interval_seconds: float = 2.0,
    start_time: float = 0.0,
    output_dir: Optional[str] = "~/Downloads",
    filename_prefix: str = "screenshot"
) -> List[ScreenshotInfo]:
    """
    Convenience function to extract screenshots from a video file.
    
    Args:
        video_path: Path to the video file
        interval_seconds: Time interval between screenshots (default: 2.0)
        start_time: Start time in seconds (default: 0.0)
        output_dir: Directory to save screenshots (default: ~/Downloads)
        filename_prefix: Prefix for saved files
        
    Returns:
        List of ScreenshotInfo objects
    """
    with VideoScreenshotExtractor(video_path) as extractor:
        return extractor.extract_screenshots(
            interval_seconds=interval_seconds,
            start_time=start_time,
            output_dir=output_dir,
            filename_prefix=filename_prefix
        )
