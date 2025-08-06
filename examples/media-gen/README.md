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

**Linux/macOS:**
```bash
./setup_env.sh
```

**Windows:**
```cmd
setup_env.bat
```

**Manual Setup:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate.bat
pip install -r requirements.txt
```

## File Structure

```
media-gen/
├── tools/                          # All tool implementations
│   ├── __init__.py                # Package exports
│   ├── media_gen_tool_base.py     # Abstract base classes
│   ├── dummy_image_gen.py         # Dummy image generation tool
│   └── dummy_video_gen.py         # Dummy video generation tool
├── tests/                         # Test suite
│   └── test_dummy_media_gen.py    # Comprehensive tests
├── example_usage.py               # Usage examples
├── requirements.txt               # Dependencies
├── setup_env.sh                   # Linux/macOS setup script
├── setup_env.bat                  # Windows setup script
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

## Usage

```python
from tools import DummyImageGen, DummyVideoGen

# Initialize tools
image_gen = DummyImageGen()
video_gen = DummyVideoGen()

# Generate media
image_result = image_gen.run({"prompt": "A beautiful sunset"})
video_result = video_gen.run({"prompt": "A butterfly emerging"})
```

## Running Examples

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate.bat

# Run example
python example_usage.py

# Run tests
cd tests && python test_dummy_media_gen.py
``` 