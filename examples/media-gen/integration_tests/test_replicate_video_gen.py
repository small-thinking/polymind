#!/usr/bin/env python3
"""
Simple script to generate videos using Replicate's WAN 2.2 i2v fast model.

Usage:
    python integration_tests/test_replicate_video_gen.py [image_path] [prompt] \
        [--timeout SECONDS] [--progress-interval SECONDS]

Examples:
    python integration_tests/test_replicate_video_gen.py
    python integration_tests/test_replicate_video_gen.py test_image.png \
        "animals playing football"
    python integration_tests/test_replicate_video_gen.py /path/to/image.jpg \
        "a magical forest scene"
    python integration_tests/test_replicate_video_gen.py test_image.png \
        "magical scene" --timeout 300 --progress-interval 10

Requirements:
- REPLICATE_API_TOKEN environment variable set
- Default test image: integration_tests/test_image.png
"""

import os
import sys

from dotenv import load_dotenv
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from tools.replicate_video_gen import ReplicateVideoGen


def main():
    """Generate a video from an image and prompt using Replicate."""
    # Check for API token
    if not os.getenv("REPLICATE_API_TOKEN"):
        print("❌ REPLICATE_API_TOKEN not found in environment variables")
        print("Please set: export REPLICATE_API_TOKEN='your_token_here'")
        sys.exit(1)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        # If it's a relative path, make it relative to the script directory
        if not Path(image_path).is_absolute():
            image_path = Path(__file__).parent / image_path
    else:
        # Default to test image
        image_path = Path(__file__).parent / "test_image.png"
    
    if len(sys.argv) > 2:
        prompt = sys.argv[2]
    else:
        # Default prompt
        prompt = "the animals standup and start playing football"
    
    # Parse optional timeout and progress interval
    timeout = 600  # 10 minutes default
    progress_interval = 5  # 5 seconds default
    
    # Simple argument parsing for timeout and progress interval
    for i, arg in enumerate(sys.argv[3:], 3):
        if arg == "--timeout" and i + 1 < len(sys.argv):
            timeout = int(sys.argv[i + 1])
        elif arg == "--progress-interval" and i + 1 < len(sys.argv):
            progress_interval = int(sys.argv[i + 1])
    
    # Validate image path
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)
    
    print(f"🎬 Generating video from: {image_path}")
    print(f"📝 Prompt: {prompt}")
    print("📁 Output: ~/Downloads/polymind_video_generation/")
    print("-" * 60)
    
    # Initialize and run video generation
    video_gen = ReplicateVideoGen()
    
    # Debug: Check if image exists and get its size
    image_path_obj = Path(image_path)
    if image_path_obj.exists():
        size_mb = image_path_obj.stat().st_size / (1024 * 1024)
        print(f"📏 Input image size: {size_mb:.2f} MB")
    else:
        print(f"❌ Image file not found: {image_path}")
        sys.exit(1)
    
    # Expand the output folder path
    output_folder = os.path.expanduser("~/Downloads/polymind_video_generation")
    
    try:
        result = video_gen.run({
            "image": str(image_path),
            "prompt": prompt,
            "output_folder": output_folder,
            "output_format": "mp4",
            "timeout": timeout,
            "progress_interval": progress_interval
        })
        
        if result["video_path"]:
            print("✅ Video generated successfully!")
            print(f"📁 Saved to: {result['video_path']}")
            
            # Show file size if available
            video_path = Path(result["video_path"])
            if video_path.exists():
                size_mb = video_path.stat().st_size / (1024 * 1024)
                print(f"📏 File size: {size_mb:.1f} MB")
        else:
            print(f"❌ Generation failed: {result['generation_info']}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 