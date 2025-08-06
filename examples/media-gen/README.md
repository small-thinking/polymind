# Media Generation Tools

A collection of abstract base classes and concrete implementations for media generation tools that can be integrated into Polymind agents. This example demonstrates how to create image and video generation tools with a consistent API.

## Summary

This package provides:
- Abstract base classes for image and video generation tools
- Dummy implementations for testing and development
- Consistent parameter specifications across all tools
- Easy integration with Polymind agents

## Setup

### Prerequisites
- Python 3.10+
- Polymind framework (installed as a third-party library)

### Quick Setup

**Linux (recommended):**
```bash
python setup.py
```

**Manual Setup:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp env.example .env
```

### Environment Configuration

The setup script automatically creates a `.env` file from the template. You'll need to edit it with your actual API keys:

**Required API keys:**
- `OPENAI_API_KEY`: For DALL-E image generation
- `REPLICATE_API_TOKEN`: For various AI models

**Note:** The `.env` file is automatically ignored by git to keep your keys secure.

## File Structure

```
media-gen/
├── tools/                          # All tool implementations
│   ├── __init__.py                # Package exports
│   ├── media_gen_tool_base.py     # Abstract base classes
│   ├── dummy_image_gen.py         # Dummy image generation tool
│   ├── dummy_video_gen.py         # Dummy video generation tool
│   └── image_understanding_tool.py # Image understanding tool
├── tests/                         # Test suite
│   ├── test_dummy_media_gen.py    # Comprehensive tests
│   └── test_image_understanding.py # Image understanding tests
├── integration_tests/             # Integration tests (manual)
│   ├── test_image_understanding.py # Real API integration test
│   └── README.md                  # Integration test documentation

├── env.example                    # Environment variables template
├── setup.py                       # Unified setup script (all platforms)
├── example_usage.py               # Usage examples
├── requirements.txt               # Dependencies
├── __init__.py                    # Main package exports
└── README.md                      # This file
```

## Creating New Tools

### Image Generation Tool

```python
from tools import ImageGenerationTool

class MyImageGen(ImageGenerationTool):
    def run(self, input: dict) -> dict:
        prompt = input.get("prompt", "")
        aspect_ratio = input.get("aspect_ratio", "4:3")
        output_format = input.get("output_format", "jpg")
        
        # Your image generation logic here
        # ...
        
        return {
            "image_path": "/path/to/generated/image.jpg",
            "generation_info": {"model": "my-model"}
        }
```

**Parameters:**
- `prompt` (str, required): Text description
- `aspect_ratio` (str, optional, default: "4:3"): Image aspect ratio
- `output_format` (str, optional, default: "jpg"): Output format

### Video Generation Tool

```python
from tools import VideoGenerationTool

class MyVideoGen(VideoGenerationTool):
    def run(self, input: dict) -> dict:
        prompt = input.get("prompt", "")
        num_frames = input.get("num_frames", 81)
        resolution = input.get("resolution", "480p")
        image = input.get("image", None)
        
        # Your video generation logic here
        # ...
        
        return {
            "video_path": "/path/to/generated/video.mp4",
            "generation_info": {"model": "my-model"}
        }
```

**Parameters:**
- `prompt` (str, required): Text description
- `num_frames` (int, optional, default: 81): Number of frames
- `resolution` (str, optional, default: "480p"): Video resolution
- `image` (str, optional): URI of starting image

### Image Understanding Tool

```python
from tools import ImageUnderstandingTool

# Initialize the tool
image_tool = ImageUnderstandingTool()

# Analyze an image from URL
result = image_tool.run({
    "prompt": "What objects do you see in this image?",
    "images": ["https://example.com/image.jpg"],
    "return_json": False
})

# Analyze with JSON response
result = image_tool.run({
    "prompt": "Analyze this image and return JSON with 'objects' and 'mood' fields",
    "images": ["path/to/local/image.jpg"],
    "return_json": True,
    "max_tokens": 500
})

# Generate image generation prompt
result = image_tool.run({
    "prompt": "Analyze this image and create a detailed image generation prompt that could be used to recreate this image. Include specific details about objects, characters, setting, lighting, mood, style, composition, colors, and textures.",
    "images": ["path/to/local/image.jpg"],
    "max_tokens": 600
})
```

**Parameters:**
- `prompt` (str, optional, default: "What's in this image?"): Analysis prompt
- `images` (List[str], required): List of image paths or URLs
- `return_json` (bool, optional, default: False): Return JSON response
- `max_tokens` (int, optional, default: 1000): Maximum response tokens


**Features:**
- Supports both local image files and image URLs
- Automatic base64 encoding for local images
- Optional JSON response format for structured output
- Configurable token limits
- Comprehensive error handling


## Testing

### Unit Tests
Run the standard unit tests:
```bash
cd tests && python test_dummy_media_gen.py
python test_image_understanding.py
```

### Integration Tests
For real API testing with actual images:
```bash
python integration_tests/test_image_understanding.py
```

**Features:**
- Generates image generation prompt for test image
- Uses local test image (`test_image.png`)
- Comprehensive error handling

**Note:** Integration tests require:
- Valid OpenAI API key in `.env` file
- Internet connection
- Test image file in `integration_tests/` folder

## Usage

```python
from tools import DummyImageGen, DummyVideoGen
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Check configuration status
print(f"OpenAI API Key: {'✓ Available' if os.getenv('OPENAI_API_KEY') else '✗ Missing'}")
print(f"Replicate API Token: {'✓ Available' if os.getenv('REPLICATE_API_TOKEN') else '✗ Missing'}")

# Initialize tools
image_gen = DummyImageGen()
video_gen = DummyVideoGen()
image_understanding = ImageUnderstandingTool()

# Generate media
image_result = image_gen.run({"prompt": "A beautiful sunset"})
video_result = video_gen.run({"prompt": "A butterfly emerging"})

# Analyze images
analysis_result = image_understanding.run({
    "prompt": "What's in this image?",
    "images": ["https://example.com/image.jpg"]
})
```

## Running Examples

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate.bat

# Run examples
python example_usage.py

# Run tests
cd tests && python test_dummy_media_gen.py
python test_image_understanding.py

# Run integration tests (requires API key)
python integration_tests/test_image_understanding.py
``` 