#!/usr/bin/env python3
"""
Integration test for video screenshot extraction utilities.

This script demonstrates video screenshot extraction functionality using the test video.
It extracts screenshots every 10 seconds and saves them to ~/Downloads.

Usage:
    python integration_tests/test_video_utils.py

Requirements:
- OpenCV (opencv-python)
- Test video file: integration_tests/test_video.mp4
"""

import os
import sys

from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.video_utils import VideoScreenshotExtractor, extract_screenshots


def main():
    """Extract screenshots from the test video every 10 seconds."""
    print("🎬 Video Screenshot Extraction Test")
    print("=" * 50)
    
    # Path to test video
    test_video_path = Path(__file__).parent / "test_video.mp4"
    
    if not test_video_path.exists():
        print(f"❌ Test video not found at: {test_video_path}")
        print("Please ensure test_video.mp4 exists in the integration_tests directory")
        return
    
    print(f"✅ Test video found: {test_video_path}")
    print(f"📏 File size: {test_video_path.stat().st_size:,} bytes")
    
    # Get video properties
    with VideoScreenshotExtractor(str(test_video_path)) as extractor:
        print(f"🎥 Video properties:")
        print(f"   - FPS: {extractor.fps:.2f}")
        print(f"   - Duration: {extractor.duration:.2f} seconds")
        print(f"   - Frame count: {extractor.frame_count:,}")
        print()
    
    # Extract screenshots every 5 seconds, starting from 1 second
    print("🔄 Extracting screenshots every 5 seconds (starting from 1s)...")
    print("-" * 50)
    
    try:
        screenshots = extract_screenshots(
            video_path=str(test_video_path),
            interval_seconds=5.0,
            start_time=1.0,  # Start from 1 second instead of 0
            output_dir="~/Downloads/polymind_video_screenshots",
            filename_prefix="test_video_5s"
        )
        
        print(f"✅ Extracted {len(screenshots)} screenshots")
        for i, ss in enumerate(screenshots):
            print(f"   {i+1}. Frame {ss.frame_number} at {ss.timestamp_str}")
        print()
        
    except Exception as e:
        print(f"❌ Screenshot extraction failed: {e}")
        return
    
    # Summary
    print("📊 Summary")
    print("=" * 50)
    downloads_dir = Path.home() / "Downloads" / "polymind_video_screenshots"
    
    if downloads_dir.exists():
        screenshot_files = list(downloads_dir.glob("test_video_10s_*.jpg"))
        print(f"✅ Screenshots saved to: {downloads_dir}")
        print(f"📁 Files created: {len(screenshot_files)}")
        
        if screenshot_files:
            print(f"📄 File pattern: test_video_10s_*.jpg")
            print(f"📏 Sample file size: {screenshot_files[0].stat().st_size:,} bytes")
    else:
        print("❌ No screenshots were created")
    
    print("\n🎯 Test completed!")
    print("💡 You can now view the extracted screenshots in your Downloads folder")


if __name__ == "__main__":
    main() 
