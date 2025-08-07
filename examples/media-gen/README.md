# Media Generation Pipeline

A command-line tool for regenerating images using AI-powered image understanding and generation.

## Overview

The media regeneration pipeline analyzes an original image and generates a new image based on user interests and preferences. It uses a two-step process:

1. **Image Understanding**: Analyzes the original image using AI vision capabilities
2. **Image Generation**: Creates a new image based on the analysis and user preferences

## Quick Start

### Prerequisites

1. **Environment Setup**: Create a `.env` file with your API keys:
   ```bash
   # Copy the example file
   cp env.example .env
   
   # Edit .env with your actual API keys
OPENAI_API_KEY=your_openai_api_key_here
REPLICATE_API_TOKEN=your_replicate_api_token_here

# Note: You only need the API key for the generator you plan to use
   ```

2. **Dependencies**: Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Basic Usage

```bash
python media_gen_pipeline.py --image-path <image_path> --user-interests <user_interests>
```

**Example:**
```bash
python media_gen_pipeline.py --image-path my_image.jpg --user-interests "make it more vibrant and modern"
```

### Command Line Options

- `--image-path`: Path to the original image file (supports ~ for home directory) (required)
- `--user-interests`: User preferences for regeneration (required)
- `--output-folder`: Output folder for generated image (default: `~/Downloads`)
- `--aspect-ratio`: Aspect ratio for generated image (default: `1:1`)
- `--output-format`: Output format for generated image (default: `png`)
- `--generator`: Image generation service to use (choices: `openai`, `replicate`, default: `openai`)
- `--debug`: Enable debug output to see prompts used in each step

**Full Example:**
```bash
python media_gen_pipeline.py \
  --image-path ~/Pictures/my_image.jpg \
  --user-interests "make it more artistic with vibrant colors" \
  --output-folder ~/Desktop/generated \
  --aspect-ratio 16:9 \
  --output-format jpg \
  --generator replicate \
  --debug
```

## Examples

### Basic Regeneration
```bash
# Regenerate with simple preferences (saves to ~/Downloads by default)
python media_gen_pipeline.py --image-path photo.jpg --user-interests "make it more vibrant"
```

### Using Replicate Generator
```bash
# Use Replicate for image generation
python media_gen_pipeline.py --image-path landscape.jpg --user-interests "convert to watercolor style" --generator replicate
```

### Using Home Directory Paths
```bash
# Use ~ for home directory paths
python media_gen_pipeline.py --image-path ~/Pictures/landscape.jpg --user-interests "convert to watercolor style"
```

### Custom Output Location
```bash
# Save to custom location
python media_gen_pipeline.py --image-path portrait.jpg --user-interests "make it modern" --output-folder ~/Desktop
```

### Artistic Transformation
```bash
# Transform to artistic style
python media_gen_pipeline.py --image-path landscape.jpg --user-interests "convert to watercolor painting style"
```

### Style Enhancement
```bash
# Enhance with specific style
python media_gen_pipeline.py --image-path portrait.jpg --user-interests "make it more modern and professional"
```

## Output

The tool outputs comprehensive information about the regeneration process with clear formatting:

```
============================================================
🎨 MEDIA REGENERATION COMPLETE
============================================================
📁 Image stored at: /Users/username/Downloads/openai_generated_image_20241201_143022.png
   📂 Relative to Downloads: openai_generated_image_20241201_143022.png

🔍 Image Analysis:
   This image shows a beautiful landscape with mountains in the background, featuring vibrant colors and dramatic lighting. The scene includes rolling hills, a clear blue sky, and natural elements that create a serene atmosphere.

🎯 Final Generation Prompt:
   This image shows a beautiful landscape with mountains in the background, featuring vibrant colors and dramatic lighting. The scene includes rolling hills, a clear blue sky, and natural elements that create a serene atmosphere.

   User preferences: make it more vibrant and modern

📊 Generation Info:
   model: gpt-4o-mini
   tokens_used: 150
   generation_time: 2.3s
============================================================
```

**Output includes:**
- **📁 Image location**: Full path where the image is stored
- **📂 Relative path**: Simplified path relative to Downloads folder (if applicable)
- **🔍 Image analysis**: The analysis of the original image
- **🎯 Final generation prompt**: Combined analysis and user interests used for generation (from metadata)
- **📊 Generation metadata**: Additional information about the generation process (model, tokens, timing, etc.)

## Path Support

The tool supports path expansion for convenience:
- **Home directory**: Use `~` to reference your home directory
- **Examples**: 
  - `~/Pictures/photo.jpg` → `/Users/username/Pictures/photo.jpg`
  - `~/Downloads` → `/Users/username/Downloads`
  - `~/.config` → `/Users/username/.config`

## Error Handling

The tool includes comprehensive error handling:
- Validates that the input image exists (with path expansion)
- Provides clear error messages for missing files or API issues
- Creates output directories automatically
- Supports path expansion for both input and output paths

## Architecture

The pipeline uses a modular design with two main components:

### Core Pipeline (`pipeline.py`)
- Generic pipeline infrastructure
- Configurable input/output mappings
- Extensible for future media types

### Media Regeneration (`media_gen_pipeline.py`)
- Specialized for image regeneration
- Command-line interface
- Simple two-parameter API
- Path expansion support

## Video Understanding

The media generation framework includes video understanding capabilities that analyze videos by extracting screenshots and generating coherent image generation prompts for each scene. This is useful for creating video-to-image pipelines where each scene can be regenerated as an image.

### Video Understanding Example

```python
from tools.video_understanding_tool import VideoUnderstandingTool

# Initialize the video understanding tool
video_understanding = VideoUnderstandingTool()

# Analyze video and generate image prompts
result = video_understanding.run({
    "video_path": "path/to/your/video.mp4",
    "user_preference": "Create cinematic image prompts with dramatic lighting",
    "screenshot_interval": 2.0,
    "output_dir": "~/Downloads/video_screenshots"
})

# Access the generated prompts
for i, prompt in enumerate(result["image_prompts"]):
    print(f"Scene {i+1}: {prompt}")
    print(f"Description: {result['scene_descriptions'][i]}")
```

### Video Understanding Parameters

- **video_path**: Path to the video file to analyze (required)
- **user_preference**: User's preference for generated image prompts (optional)
- **screenshot_interval**: Time interval between screenshots in seconds (optional, default: 2.0)
- **output_dir**: Directory to save extracted screenshots (optional, default: ~/Downloads)
- **max_tokens**: Maximum tokens in response (optional, default: 2000)

### Testing Video Understanding

Run the video understanding integration test:

```bash
python integration_tests/test_video_understanding_tool.py
```

## Video Generation

### Video Generation Example

```python
from tools.replicate_video_gen import ReplicateVideoGen

# Initialize the video generation tool
video_gen = ReplicateVideoGen()

# Generate video from image and text prompt
result = video_gen.run({
    "image": "path/to/your/image.jpg",
    "prompt": "A serene landscape with gentle movement and natural lighting",
    "output_folder": "~/Downloads/polymind_videos",
    "output_format": "mp4"
})

print(f"Video saved to: {result['video_path']}")
```

### Video Generation Parameters

- **image**: Image path, URL, or data URI (required)
- **prompt**: Text description of the desired video (required)
- **output_folder**: Folder path where to save the video (optional, default: "~/Downloads")
- **output_format**: Output format (optional, default: "mp4")
- **model**: Replicate model to use (optional, overrides default)

### Testing Video Generation

Run the video generation integration test:

```bash
python integration_tests/test_replicate_video_gen.py
```

## Future Extensions

The modular design allows easy extension to other media types:
- **Video Understanding**: Add video analysis capabilities
- **Multi-modal**: Support for text, audio, and other media

## File Structure

```
media-gen/
├── media_gen_pipeline.py    # Main command-line tool
├── pipeline.py              # Core pipeline infrastructure
├── .env                     # API keys (create from env.example)
├── env.example              # Environment variables template
├── tools/                   # Media generation tools
│   ├── image_understanding_tool.py
│   ├── video_understanding_tool.py
│   ├── openai_image_gen.py
│   ├── replicate_image_gen.py
│   ├── replicate_video_gen.py
│   ├── dummy_image_gen.py
│   ├── dummy_video_gen.py
│   └── media_gen_tool_base.py
├── tests/                   # Test files
│   └── test_replicate_video_gen.py
├── integration_tests/       # Integration test files and examples
│   ├── test_replicate_video_gen.py
│   └── test_video_understanding_tool.py
└── ~/Downloads/            # Default output location
```

## Testing

Test with the provided test image:

**Using OpenAI (default):**
```bash
python media_gen_pipeline.py --image-path integration_tests/test_image.png --user-interests "enhance the visual appeal"
```

**Using Replicate:**
```bash
python media_gen_pipeline.py --image-path integration_tests/test_image.png --user-interests "enhance the visual appeal" --generator replicate
```