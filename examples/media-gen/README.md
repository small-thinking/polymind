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
│   ├── openai_image_gen.py        # OpenAI image generation tool
│   ├── replicate_image_gen.py     # Replicate image generation tool
│   ├── dummy_video_gen.py         # Dummy video generation tool
│   └── image_understanding_tool.py # Image understanding tool
├── tests/                         # Test suite
│   ├── test_dummy_media_gen.py    # Comprehensive tests
│   ├── test_openai_image_gen.py   # OpenAI image generation tests
│   ├── test_replicate_image_gen.py # Replicate image generation tests
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
        output_folder = input.get("output_folder", "/tmp")
        
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
- `output_folder` (str, optional, default: "~/Downloads"): Folder path where to save the generated image

### OpenAI Image Generation Tool

```python
from tools import OpenAIImageGen

# Initialize the tool
image_gen = OpenAIImageGen()

# Basic usage
result = image_gen.run({
    "prompt": "A gray tabby cat hugging an otter with an orange scarf",
    "output_folder": "./generated_images"
})

# Advanced usage with custom parameters
result = image_gen.run({
    "prompt": "A futuristic cityscape at sunset with flying cars",
    "size": "1024x1536",
    "quality": "high",
    "output_format": "png",
    "compression": 90,
    "background": "opaque",
    "output_folder": "./generated_images"
})
```

**Parameters:**
- `prompt` (str, required): Text description of the desired image
- `output_folder` (str, optional, default: "~/Downloads"): Folder path where to save the generated image
- `size` (str, optional, default: "1024x1024"): Image dimensions
- `quality` (str, optional, default: "low"): Rendering quality (low, medium, high)
- `output_format` (str, optional, default: "png"): Output format
- `compression` (int, optional, default: 80): Compression level 0-100%
- `background` (str, optional, default: "opaque"): Transparent or opaque

**Features:**
- Uses OpenAI's gpt-4o-mini model with image generation capabilities
- Supports various image parameters (size, quality, format, compression, background)
- Automatic directory creation for output paths
- Comprehensive error handling
- Integrates seamlessly with Polymind framework

### Replicate Image Generation Tool

```python
from tools import ReplicateImageGen

# Initialize the tool with default model (WAN 2.2)
image_gen = ReplicateImageGen()

# Basic usage
result = image_gen.run({
    "prompt": "A cinematic cat portrait with golden hour lighting",
    "output_folder": "./generated_images"
})

# Advanced usage with custom parameters
result = image_gen.run({
    "prompt": "A cinematic, photorealistic medium shot of a cat",
    "seed": 246764,
    "aspect_ratio": "4:3",
    "model": "stability-ai/sdxl"
})
```

**Parameters:**
- `prompt` (str, required): Text description of the desired image
- `output_folder` (str, optional, default: "~/Downloads"): Folder path where to save the generated image
- `seed` (int, optional): Random seed for reproducible results
- `aspect_ratio` (str, optional, default: "4:3"): Image aspect ratio
- `output_format` (str, optional, default: "jpeg"): Output format
- `model` (str, optional): Replicate model to use (overrides default)

**Features:**
- Uses Replicate's API with various image generation models
- Supports models like WAN 2.2, Stable Diffusion XL, and others
- Reproducible results with seed parameter
- Automatic directory creation for output paths
- Comprehensive error handling
- Integrates seamlessly with Polymind framework

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
from tools import DummyImageGen, OpenAIImageGen, ReplicateImageGen, DummyVideoGen
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Check configuration status
print(f"OpenAI API Key: {'✓ Available' if os.getenv('OPENAI_API_KEY') else '✗ Missing'}")
print(f"Replicate API Token: {'✓ Available' if os.getenv('REPLICATE_API_TOKEN') else '✗ Missing'}")

# Initialize tools
image_gen = DummyImageGen()
openai_image_gen = OpenAIImageGen()
replicate_image_gen = ReplicateImageGen()
video_gen = DummyVideoGen()
image_understanding = ImageUnderstandingTool()

# Generate media
image_result = image_gen.run({"prompt": "A beautiful sunset"})
openai_result = openai_image_gen.run({
    "prompt": "A beautiful sunset over mountains",
    "output_folder": "./generated_images"
})
replicate_result = replicate_image_gen.run({
    "prompt": "A cinematic cat portrait",
    "seed": 12345,
    "aspect_ratio": "4:3"
})
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
python test_openai_image_gen.py
python test_replicate_image_gen.py
python test_image_understanding.py

# Run integration tests (requires API key)
python integration_tests/test_image_understanding.py
``` 