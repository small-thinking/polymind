"""
Integration test for Replicate image generation tool.

This script tests the Replicate image generation tool with real API calls.
It should be run manually when you want to test the actual image generation
functionality.

Requirements:
- Valid Replicate API token in environment variables
- Internet connection
- Replicate API access
"""

import os
import sys

from dotenv import load_dotenv
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from tools.replicate_image_gen import ReplicateImageGen
except ImportError as e:
    if "replicate" in str(e):
        print("❌ Replicate package not installed. Please install it with:")
        print("   pip install replicate")
        sys.exit(1)
    else:
        raise


def main():
    """Generate a single image with the specified prompt."""
    print("🚀 Replicate Image Generation Integration Test")
    print("=" * 60)
    
    # Check if Replicate API token is available
    if not os.getenv("REPLICATE_API_TOKEN"):
        print("❌ REPLICATE_API_TOKEN environment variable not set")
        print("Please set your Replicate API token to run integration tests")
        return
    
    token = os.getenv('REPLICATE_API_TOKEN')
    print(f"✅ Replicate API token found: {token[:10]}...")
    
    # Create Downloads directory if it doesn't exist
    downloads_dir = Path.home() / "Downloads" / "polymind_generated_images"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Output directory: {downloads_dir.absolute()}")
    
    # Initialize the tool
    image_gen = ReplicateImageGen()
    
    # The specific prompt from the example
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
            "seed": 246764,
            "aspect_ratio": "4:3",
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
            if "api token" in error_msg.lower():
                print("\n💡 To fix this issue:")
                print("1. Check that your REPLICATE_API_TOKEN is correct")
                print("2. Ensure you have sufficient credits in your Replicate account")
                print("3. Verify your account has access to the model")
            elif "model" in error_msg.lower():
                print("\n💡 To fix this issue:")
                print("1. Check that the model name is correct")
                print("2. Ensure the model is publicly available")
                print("3. Verify your account has access to the model")
            
    except Exception as e:
        print(f"\n❌ Integration test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Run the integration test
    main() 