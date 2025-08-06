#!/usr/bin/env python3
"""
Simple integration test for ImageUnderstandingTool.

This test uses the test_image.png file to generate an image generation prompt.
It requires OPENAI_API_KEY in the .env file.

Run with: python integration_tests/test_image_understanding.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add the parent directory to the path to import the tool
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.image_understanding_tool import ImageUnderstandingTool


def main():
    """Run the integration test."""
    print("=== Image Understanding Integration Test ===")
    print("This test will analyze test_image.png and generate an image generation prompt.\n")
    
    # Load environment variables
    load_dotenv()
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment.")
        print("Please set it in your .env file.")
        return
    
    # Get test image path
    test_image_path = Path(__file__).parent / "test_image.png"
    
    if not test_image_path.exists():
        print(f"❌ Error: Test image not found at {test_image_path}")
        return
    
    print(f"✅ Found test image: {test_image_path}")
    print(f"✅ API key available: {api_key[:8]}...")
    print()
    
    # Initialize tool
    tool = ImageUnderstandingTool()
    
    # Define the prompt for image generation
    prompt = (
        "Analyze this image and create an image generation prompt that could be used "
        "to recreate this image as similar as possible. "
        "Include specific details about objects, characters, setting, "
        "lighting, mood, image style, composition, colors, and textures. The prompt should be "
        "less than 100 words."
    )
    
    print("📝 Prompt: Generate image generation prompt")
    print("🔄 Calling OpenAI API...")
    
    try:
        result = tool.run({
            "prompt": prompt,
            "images": [str(test_image_path)],
            "return_json": False,
            "max_tokens": 600
        })
        
        print("✅ Analysis completed successfully!")
        print("\n📋 Generated Image Generation Prompt:")
        print("-" * 50)
        print(result["analysis"])
        print("-" * 50)
        
        print("\n📊 Metadata:")
        print(f"  Model: {result['metadata']['model']}")
        print(f"  Total tokens: {result['metadata']['tokens_used']}")
        print(f"  Prompt tokens: {result['metadata']['prompt_tokens']}")
        print(f"  Completion tokens: {result['metadata']['completion_tokens']}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return
    
    print("\n✅ Integration test completed successfully!")


if __name__ == "__main__":
    main() 