"""
Example usage of the media generation tools.

This script demonstrates how to use the DummyImageGen and DummyVideoGen tools
with the new parameter specifications and environment variable configuration.
"""

import os
from dotenv import load_dotenv
from tools import DummyImageGen, DummyVideoGen


def check_config_status():
    """Check and display configuration status."""
    print("Configuration Status:")
    print("=" * 40)
    
    # Load environment variables
    load_dotenv()
    
    # Check API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    replicate_token = os.getenv("REPLICATE_API_TOKEN")
    
    print(f"OpenAI API Key: {'✓ Available' if openai_key else '✗ Missing'}")
    print(f"Replicate API Token: "
          f"{'✓ Available' if replicate_token else '✗ Missing'}")
    
    # Show configuration
    default_image = os.getenv('DEFAULT_IMAGE_MODEL', 'dall-e-3')
    default_video = os.getenv('DEFAULT_VIDEO_MODEL', 'stable-video-diffusion')
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    
    print(f"\nDefault Image Model: {default_image}")
    print(f"Default Video Model: {default_video}")
    print(f"Log Level: {log_level}")
    
    if not openai_key or not replicate_token:
        print("\nMissing API keys. Please check your .env file.")


def main():
    """Demonstrate the media generation tools."""
    print("Media Generation Tools Example")
    print("=" * 40)
    
    # Check configuration status
    print("\nConfiguration Status:")
    check_config_status()
    
    # Initialize tools
    image_gen = DummyImageGen()
    video_gen = DummyVideoGen()
    
    # Example 1: Generate an image with default parameters
    print("\n1. Generating image with defaults...")
    image_result = image_gen.run({
        "prompt": "A serene mountain landscape at dawn"
    })
    print(f"   Image path: {image_result['image_path']}")
    print(f"   Info: {image_result['generation_info']}")
    
    # Example 2: Generate an image with custom parameters
    print("\n2. Generating image with custom parameters...")
    image_result = image_gen.run({
        "prompt": "A futuristic city skyline",
        "aspect_ratio": "16:9",
        "output_format": "png"
    })
    print(f"   Image path: {image_result['image_path']}")
    print(f"   Info: {image_result['generation_info']}")
    
    # Example 3: Generate a video with default parameters
    print("\n3. Generating video with defaults...")
    video_result = video_gen.run({
        "prompt": "A butterfly emerging from a cocoon"
    })
    print(f"   Video path: {video_result['video_path']}")
    print(f"   Info: {video_result['generation_info']}")
    
    # Example 4: Generate a video with custom parameters
    print("\n4. Generating video with custom parameters...")
    video_result = video_gen.run({
        "prompt": "A cinematic flythrough of a space station",
        "num_frames": 120,
        "resolution": "720p",
        "image": "https://example.com/space_station.jpg"
    })
    print(f"   Video path: {video_result['video_path']}")
    print(f"   Info: {video_result['generation_info']}")
    
    print("\n" + "=" * 40)
    print("Example completed successfully!")


if __name__ == "__main__":
    main() 