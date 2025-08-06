"""
Integration test for OpenAI image generation tool.

This script tests the OpenAI image generation tool with a real API call.
It generates a single image with a specific prompt and saves it to ~/Downloads.

Requirements:
- Valid OpenAI API key in environment variables
- Internet connection
- OpenAI API access
"""

import os
import sys

from dotenv import load_dotenv
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tools.openai_image_gen import OpenAIImageGen


def main():
    """Generate a single image with the specified prompt."""
    print("🚀 OpenAI Image Generation Integration Test")
    print("=" * 60)
    
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key to run integration tests")
        return
    
    print(f"✅ OpenAI API key found: {os.getenv('OPENAI_API_KEY')[:10]}...")
    
    # Create Downloads directory if it doesn't exist
    downloads_dir = Path.home() / "Downloads" / "polymind_generated_images"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Output directory: {downloads_dir.absolute()}")
    
    # Initialize the tool
    image_gen = OpenAIImageGen()
    
    # The specific prompt provided by the user
    prompt = (
        "Create a whimsical scene inside a modern subway train featuring a "
        "fluffy, light brown rabbit and a chubby, soft gray penguin sitting "
        "side by side on blue plastic seats. The background includes tall "
        "city buildings visible through a large window and a colorful framed "
        "picture on the wall. Soft, natural lighting filtering through the "
        "window enhances the cheerful atmosphere. The image should be "
        "hyper-realistic with fine textures on the animals' fur and "
        "feathers, emphasizing their cuteness while maintaining a playful "
        "and friendly mood."
    )
    
    print("\n🎨 Generating image with prompt:")
    print(f"'{prompt[:100]}...'")
    print()
    
    try:
        # Generate the image
        result = image_gen.run({
            "prompt": prompt,
            "size": "1024x1024",
            "quality": "high",
            "output_format": "png",
            "output_folder": str(downloads_dir)
        })
        
        print(f"Result: {result}")
        
        if result["image_path"] and os.path.exists(result["image_path"]):
            file_size = os.path.getsize(result["image_path"])
            print("✅ Image generated successfully!")
            print(f"📁 Saved to: {result['image_path']}")
            print(f"📏 File size: {file_size:,} bytes")
            print(f"🎯 Generation info: {result['generation_info']}")
        else:
            print("❌ Image generation failed")
            error_msg = result.get('generation_info', {}).get('error', 'Unknown error')
            print(f"Error: {error_msg}")
            
            # Check for specific error types and provide helpful guidance
            if "organization must be verified" in error_msg.lower():
                print("\n💡 To fix this issue:")
                print("1. Go to: https://platform.openai.com/settings/organization/general")
                print("2. Click on 'Verify Organization'")
                print("3. Wait up to 15 minutes for access to propagate")
                print("4. Try running this test again")
            elif "api key" in error_msg.lower():
                print("\n💡 To fix this issue:")
                print("1. Check that your OPENAI_API_KEY is correct")
                print("2. Ensure you have sufficient credits in your OpenAI account")
                print("3. Verify your account has access to image generation features")
            
    except Exception as e:
        print(f"\n❌ Integration test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Run the integration test
    main() 