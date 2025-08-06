"""
Test script for the dummy media generation tools.

This script demonstrates and tests the basic functionality of the DummyImageGen
and DummyVideoGen tools. It verifies that the tools can be instantiated and
return expected dummy results without requiring actual media generation APIs.
"""

import sys
import os

# Add the parent directory to sys.path to enable imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the new tools package
from tools import DummyImageGen, DummyVideoGen


def test_dummy_image_generation():
    """Test the DummyImageGen tool functionality."""
    print("Testing DummyImageGen...")
    
    # Initialize the tool
    image_gen = DummyImageGen()
    
    # Test basic image generation with new parameters
    input_data = {
        "prompt": "A beautiful landscape with mountains and lakes",
        "aspect_ratio": "16:9",
        "output_format": "png"
    }
    
    result = image_gen.run(input_data)
    
    print(f"Tool Name: {image_gen.tool_name}")
    print(f"Generated Image Path: {result['image_path']}")
    print(f"Generation Info: {result['generation_info']}")
    
    # Verify the result structure
    assert "image_path" in result, "Missing image_path in result"
    assert "generation_info" in result, "Missing generation_info in result"
    assert isinstance(result["generation_info"], dict), "generation_info should be a dict"
    
    print("✓ DummyImageGen test passed!\n")
    return result


def test_dummy_video_generation():
    """Test the DummyVideoGen tool functionality."""
    print("Testing DummyVideoGen...")
    
    # Initialize the tool
    video_gen = DummyVideoGen()
    
    # Test basic video generation with new parameters
    input_data = {
        "prompt": "A timelapse of clouds moving across the sky",
        "num_frames": 120,
        "resolution": "720p",
        "image": "https://example.com/starting_image.jpg"
    }
    
    result = video_gen.run(input_data)
    
    print(f"Tool Name: {video_gen.tool_name}")
    print(f"Generated Video Path: {result['video_path']}")
    print(f"Generation Info: {result['generation_info']}")
    
    # Verify the result structure
    assert "video_path" in result, "Missing video_path in result"
    assert "generation_info" in result, "Missing generation_info in result"
    assert isinstance(result["generation_info"], dict), "generation_info should be a dict"
    
    print("✓ DummyVideoGen test passed!\n")
    return result


def test_tool_specs():
    """Test the input and output specifications of the tools."""
    print("Testing tool specifications...")
    
    # Test image tool specs
    image_gen = DummyImageGen()
    input_spec = image_gen.input_spec()
    output_spec = image_gen.output_spec()
    
    print(f"Image tool input spec: {len(input_spec)} parameters")
    print(f"Image tool output spec: {len(output_spec)} parameters")
    
    # Test video tool specs
    video_gen = DummyVideoGen()
    input_spec = video_gen.input_spec()
    output_spec = video_gen.output_spec()
    
    print(f"Video tool input spec: {len(input_spec)} parameters")
    print(f"Video tool output spec: {len(output_spec)} parameters")
    
    print("✓ Tool specifications test passed!\n")


def run_integration_test():
    """Run a full integration test simulating a media generation workflow."""
    print("Running integration test...")
    
    # Simulate an agent workflow that needs both image and video generation
    image_gen = DummyImageGen()
    video_gen = DummyVideoGen()
    
    # Step 1: Generate a concept image
    image_result = image_gen.run({
        "prompt": "A futuristic city skyline at sunset",
        "aspect_ratio": "4:3",
        "output_format": "jpg"
    })
    
    # Step 2: Generate a video based on the image concept  
    video_result = video_gen.run({
        "prompt": "A cinematic flythrough of a futuristic city skyline at sunset",
        "num_frames": 81,
        "resolution": "480p",
        "image": "https://example.com/concept_image.jpg"
    })
    
    print("Integration workflow completed:")
    print(f"  Image: {image_result['image_path']}")
    print(f"  Video: {video_result['video_path']}")
    
    print("✓ Integration test passed!\n")


if __name__ == "__main__":
    """Main test execution."""
    print("=" * 60)
    print("MEDIA GENERATION TOOLS TEST SUITE")
    print("=" * 60)
    print()
    
    try:
        # Run individual tool tests
        image_result = test_dummy_image_generation()
        video_result = test_dummy_video_generation()
        
        # Test tool specifications
        test_tool_specs()
        
        # Run integration test
        run_integration_test()
        
        print("=" * 60)
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("The media generation framework structure is working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        sys.exit(1)