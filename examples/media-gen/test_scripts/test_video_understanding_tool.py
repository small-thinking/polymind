#!/usr/bin/env python3
"""
Integration test for VideoUnderstandingTool.

This script tests the VideoUnderstandingTool with test_video.mp4 using either
interval-based or keyframe-based extraction methods.

Requirements:
- Valid OpenAI API key in environment variables
- test_video.mp4 file in the integration_tests directory
- Internet connection
- OpenAI API access

Run with: python integration_tests/test_video_understanding_tool.py [--interval]
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from pathlib import Path

# Add the parent directory to the path to import the tool
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.video_understanding_tool import VideoUnderstandingTool


def test_extraction(tool, test_video_path, use_interval=False):
    """Test video extraction with specified method."""
    if use_interval:
        print("📸 Testing Interval-Based Extraction")
        print("-" * 40)
        
        result = tool.run({
            "video_path": str(test_video_path),
            "user_preference": "cinematic style with dramatic lighting",
            "extraction_mode": "interval",
            "screenshot_interval": 10.0,
            "output_dir": "~/Downloads/video_understanding_interval"
        })
        
        print("✅ Interval-based analysis completed!")
        print(f"📊 Scenes: {result['metadata']['total_scenes']}")
        print(f"⏱️  Duration: {result['metadata']['video_duration']}")
        print(f"📸 Interval: {result['metadata']['screenshot_interval']}")
        
    else:
        print("🔍 Testing Keyframe-Based Extraction")
        print("-" * 40)
        
        result = tool.run({
            "video_path": str(test_video_path),
            "user_preference": "cinematic style with dramatic lighting",
            "extraction_mode": "keyframe",
            "keyframe_threshold": 25.0,
            "min_interval_frames": 15,
            "output_dir": "~/Downloads/video_understanding_keyframe"
        })
        
        print("✅ Keyframe-based analysis completed!")
        print(f"📊 Scenes: {result['metadata']['total_scenes']}")
        print(f"⏱️  Duration: {result['metadata']['video_duration']}")
        print(f"🎯 Threshold: {result['metadata']['keyframe_threshold']}")
    
    return len(result["image_prompts"])


def main():
    """Run the integration test."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Test VideoUnderstandingTool with different extraction methods"
    )
    parser.add_argument(
        "--interval",
        action="store_true",
        help="Use interval-based extraction (default: keyframe-based)"
    )
    args = parser.parse_args()
    
    extraction_method = "interval" if args.interval else "keyframe"
    
    print("=== Video Understanding Integration Test ===")
    print(f"This test will analyze test_video.mp4 using {extraction_method}-based extraction.\n")
    
    # Load environment variables
    load_dotenv()
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment.")
        print("Please set it in your .env file.")
        return
    
    # Get test video path
    test_video_path = Path(__file__).parent / "test_video.mp4"
    
    if not test_video_path.exists():
        print(f"❌ Error: Test video not found at {test_video_path}")
        print("Please place a test video file named 'test_video.mp4' in the "
              "integration_tests directory.")
        return
    
    print(f"✅ Found test video: {test_video_path}")
    print(f"✅ API key available: {api_key[:8]}...")
    print()
    
    # Initialize tool
    tool = VideoUnderstandingTool()
    
    # Test extraction method
    try:
        scene_count = test_extraction(tool, test_video_path, args.interval)
        print(f"\n📊 Results: {scene_count} scenes detected")
        print("\n✅ Integration test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return


if __name__ == "__main__":
    main()
