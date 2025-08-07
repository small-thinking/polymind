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

## Future Extensions

The modular design allows easy extension to other media types:
- **Image to Video**: Add video generation step
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
│   ├── openai_image_gen.py
│   ├── dummy_image_gen.py
│   └── media_gen_tool_base.py
├── integration_tests/       # Test files and examples
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