#!/usr/bin/env python3
"""
Test script for Replicate video generation tool using WAN 2.2 i2v fast model.

This script demonstrates how to use the ReplicateVideoGen tool to generate
videos from images and text prompts using the WAN 2.2 i2v fast model.

Usage:
    python tests/test_replicate_video_gen.py

Requirements:
- Replicate API token set in environment variables
- Test image file: tests/test_image.png
"""

import base64
import os
import sys

from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tools.replicate_video_gen import ReplicateVideoGen


def test_replicate_video_generation():
    """Test the Replicate video generation functionality."""
    print("🎬 Replicate Video Generation Test")
    print("=" * 50)
    
    # Check for Replicate API token
    if not os.getenv("REPLICATE_API_TOKEN"):
        print("❌ REPLICATE_API_TOKEN not found in environment variables")
        print("Please set your Replicate API token:")
        print("export REPLICATE_API_TOKEN='your_token_here'")
        return False
    
    # Path to test image
    test_image_path = Path(__file__).parent / "test_image.png"
    
    if not test_image_path.exists():
        print(f"❌ Test image not found at: {test_image_path}")
        print("Please ensure test_image.png exists in the tests directory")
        return False
    
    print(f"✅ Test image found: {test_image_path}")
    print(f"📏 File size: {test_image_path.stat().st_size:,} bytes")
    
    # Initialize the video generation tool
    video_gen = ReplicateVideoGen()
    
    # Test parameters
    test_prompt = (
        "Close-up shot of an elderly sailor wearing a yellow raincoat, "
        "seated on the deck of a catamaran, slowly puffing on a pipe. "
        "His cat lies quietly beside him with eyes closed, enjoying the "
        "calm. The warm glow of the setting sun bathes the scene, with "
        "gentle waves lapping against the hull and a few seabirds "
        "circling slowly above. The camera slowly pushes in, capturing "
        "this peaceful and harmonious moment."
    )
    
    print(f"📝 Test prompt: {test_prompt[:100]}...")
    print()
    
    # Generate video
    print("🔄 Generating video from image and text prompt...")
    print("-" * 50)
    
    try:
        result = video_gen.run({
            "image": str(test_image_path),
            "prompt": test_prompt,
            "output_folder": "~/Downloads/polymind_video_generation",
            "output_format": "mp4"
        })
        
        if result["video_path"]:
            print(f"✅ Video generated successfully!")
            print(f"📁 Video saved to: {result['video_path']}")
            print(f"📊 Generation info: {result['generation_info']}")
            
            # Check if file exists and get size
            video_path = Path(result["video_path"])
            if video_path.exists():
                print(f"📏 Video file size: {video_path.stat().st_size:,} bytes")
            else:
                print("⚠️  Video file not found at expected location")
            
            return True
        else:
            print(f"❌ Video generation failed: {result['generation_info']}")
            return False
            
    except Exception as e:
        print(f"❌ Video generation failed with exception: {e}")
        return False


def test_with_data_uri():
    """Test video generation using data URI for image input."""
    print("\n🔄 Testing with data URI image input...")
    print("-" * 50)
    
    # Path to test image
    test_image_path = Path(__file__).parent / "test_image.png"
    
    if not test_image_path.exists():
        print("❌ Test image not found for data URI test")
        return False
    
    # Convert image to data URI
    with open(test_image_path, 'rb') as file:
        data = base64.b64encode(file.read()).decode('utf-8')
        data_uri = f"data:application/octet-stream;base64,{data}"
    
    print(f"✅ Converted image to data URI ({len(data_uri)} chars)")
    
    # Initialize the video generation tool
    video_gen = ReplicateVideoGen()
    
    # Test parameters
    test_prompt = "A serene landscape with gentle movement and natural lighting"
    
    try:
        result = video_gen.run({
            "image": data_uri,
            "prompt": test_prompt,
            "output_folder": "~/Downloads/polymind_video_generation",
            "output_format": "mp4"
        })
        
        if result["video_path"]:
            print(f"✅ Video generated successfully with data URI!")
            print(f"📁 Video saved to: {result['video_path']}")
            return True
        else:
            print(f"❌ Video generation failed: {result['generation_info']}")
            return False
            
    except Exception as e:
        print(f"❌ Video generation failed with exception: {e}")
        return False


def main():
    """Run all video generation tests."""
    print("🎬 Replicate Video Generation Tool Tests")
    print("=" * 60)
    
    # Test 1: Basic video generation
    success1 = test_replicate_video_generation()
    
    # Test 2: Data URI input
    success2 = test_with_data_uri()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 60)
    print(f"✅ Basic video generation: {'PASS' if success1 else 'FAIL'}")
    print(f"✅ Data URI input: {'PASS' if success2 else 'FAIL'}")
    
    if success1 and success2:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    print("\n💡 Generated videos are saved to ~/Downloads/polymind_video_generation/")


if __name__ == "__main__":
    main() 