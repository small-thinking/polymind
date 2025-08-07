"""
Test script for VideoUnderstandingTool.

This script tests the VideoUnderstandingTool with a specific video file
and user preference for kapybara, cat, and football.
"""

import os
import sys

from pathlib import Path

# Add the parent directory to the path to import the tool
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.video_understanding_tool import VideoUnderstandingTool


def test_video_understanding():
    """Test video understanding with kapybara, cat, and football preference."""
    
    # Check if OpenAI API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("Please set: export OPENAI_API_KEY='your_token_here'")
        return False
    
    # Initialize the tool
    print("🔧 Initializing VideoUnderstandingTool...")
    tool = VideoUnderstandingTool(api_key=api_key)
    
    # Test video path
    test_video_path = Path(__file__).parent / "test_video.mp4"
    
    if not test_video_path.exists():
        print(f"❌ Test video not found: {test_video_path}")
        print("Please place a test video file at the above path")
        return False
    
    print(f"🎬 Analyzing video: {test_video_path}")
    print("📝 User preference: kapybara, cat, and football")
    print("📁 Output directory: ~/Downloads/video_understanding")
    print("-" * 60)
    
    try:
        # Run the video understanding tool
        result = tool.run({
            "video_path": str(test_video_path),
            "user_preference": "kapybara, cat, and football",
            "screenshot_interval": 10.0,
            "output_dir": "~/Downloads/video_understanding"
        })
        
        # Display results
        print("✅ Video analysis completed successfully!")
        print(f"📊 Total scenes analyzed: {result['metadata']['total_scenes']}")
        print(f"⏱️  Video duration: {result['metadata']['video_duration']}")
        print(f"📸 Screenshot interval: {result['metadata']['screenshot_interval']}")
        print()
        
        print("🎨 Generated Prompts:")
        print("=" * 50)
        for i, (img_prompt, vid_prompt, description) in enumerate(
            zip(result["image_prompts"], result["video_prompts"], result["scene_descriptions"]), 1
        ):
            print(f"Scene {i}: {description}")
            print(f"Image Prompt: {img_prompt}")
            print(f"Video Prompt: {vid_prompt}")
            print(f"Screenshot: {result['screenshot_paths'][i-1]}")
            print("-" * 30)
        
        print("\n📁 Screenshots saved to:")
        for path in result["screenshot_paths"]:
            print(f"  - {path}")
        
        print(f"\n🔧 Model used: {result['metadata']['model']}")
        print(f"📊 Tokens used: {result['metadata'].get('tokens_used', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False


if __name__ == "__main__":
    print("🎬 Video Understanding Tool Test")
    print("=" * 50)
    
    success = test_video_understanding()
    
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
