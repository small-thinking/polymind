"""
Video regeneration pipeline.

Command-line tool for regenerating videos:
1. Analyze original video using video understanding
2. Generate images for each scene using image generation
3. Generate videos from each image using video generation

Usage:
    python video_regen_pipeline.py --video-path <video_path> 
    --user-interests <user_interests>

Example:
    python video_regen_pipeline.py --video-path \
    ./examples/media-gen/integration_tests/test_video.mp4 \
    --user-interests "Users like cinematic style with dramatic lighting"
"""

import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, List

from pathlib import Path

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, continue without it
    pass

from pipeline import MediaGenerationPipeline, PipelineStep, PipelineStepExecutor
from tools.replicate_image_gen import ReplicateImageGen
from tools.replicate_video_gen import ReplicateVideoGen
from tools.video_understanding_tool import VideoUnderstandingTool


def expand_path(path: str) -> str:
    """Expand path with ~ to user's home directory."""
    return os.path.expanduser(path)


class VideoRegenerationPipeline(MediaGenerationPipeline):
    """
    Pipeline for regenerating videos.
    
    Workflow:
    1. Analyze original video using video understanding
    2. Generate images for each scene using image generation
    3. Generate videos from each image using video generation
    """

    def __init__(
        self,
        video_understanding_tool: VideoUnderstandingTool,
        image_generation_tool: Any,  # BaseTool type
        video_generation_tool: Any,  # BaseTool type
        name: str = "video_regeneration",
        debug: bool = False
    ):
        """
        Initialize the video regeneration pipeline.
        
        Args:
            video_understanding_tool: Tool for analyzing videos
            image_generation_tool: Tool for generating images
            video_generation_tool: Tool for generating videos
            name: Pipeline name
            debug: Enable debug output
        """
        super().__init__(name)
        self.debug = debug
        
        # Add video understanding step
        self.add_step(
            PipelineStep(
                name="video_understanding",
                tool=video_understanding_tool,
                input_mapping={
                    "original_video": "video_path",
                    "user_preferences": "user_preference",
                    "extraction_mode": "extraction_mode",
                    "screenshot_interval": "screenshot_interval",
                    "keyframe_threshold": "keyframe_threshold",
                    "min_interval_frames": "min_interval_frames",
                    "output_dir": "output_dir"
                },
                output_mapping={
                    "image_prompts": "image_prompts",
                    "video_prompts": "video_prompts",
                    "scene_descriptions": "scene_descriptions",
                    "screenshot_paths": "screenshot_paths",
                    "metadata": "video_metadata"
                }
            )
        )
        
        # Add image generation step (for each scene)
        self.add_step(
            PipelineStep(
                name="image_generation",
                tool=image_generation_tool,
                input_mapping={
                    "image_prompts": "prompt",
                    "output_folder": "output_folder",
                    "aspect_ratio": "aspect_ratio",
                    "output_format": "output_format"
                },
                output_mapping={
                    "generated_image_paths": "generated_image_paths",
                    "image_generation_info": "image_generation_info"
                },
                transform_input=self._prepare_image_generation,
                transform_output=self._extract_image_paths
            )
        )
        
        # Add video generation step (for each generated image)
        self.add_step(
            PipelineStep(
                name="video_generation",
                tool=video_generation_tool,
                input_mapping={
                    "generated_image_paths": "image",
                    "video_prompts": "prompt",
                    "output_folder": "output_folder",
                    "output_format": "output_format"
                },
                output_mapping={
                    "generated_videos": "generated_video_paths",
                    "video_generation_info": "video_generation_info"
                },
                transform_input=self._prepare_video_generation,
                transform_output=self._extract_video_paths
            )
        )
    
    def regenerate(
        self,
        video_path: str,
        user_interests: str,
        output_folder: str = "~/Downloads",
        extraction_mode: str = "interval",
        screenshot_interval: float = 10.0,
        keyframe_threshold: float = 30.0,
        min_interval_frames: int = 30,
        aspect_ratio: str = "9:16",
        output_format: str = "mp4"
    ) -> Dict[str, Any]:
        """
        Regenerate a video based on the original.
        
        Args:
            video_path: Path to the original video
            user_interests: User preferences for regeneration
            output_folder: Folder to save generated videos 
                          (default: ~/Downloads)
            extraction_mode: 'interval' or 'keyframe' (default: 'interval')
            screenshot_interval: Time interval between screenshots (for interval mode)
            keyframe_threshold: Threshold for keyframe detection (for keyframe mode)
            min_interval_frames: Minimum frames between keyframes (for keyframe mode)
            aspect_ratio: Aspect ratio for generated images
            output_format: Output format for generated videos
            
        Returns:
            Dictionary containing:
                - generated_video_paths: List of paths to generated videos
                - generated_image_paths: List of paths to generated images
                - video_analysis: Analysis from video understanding
                - generation_metadata: Additional generation info
        """
        # Create organized output structure
        expanded_output_folder = expand_path(output_folder)
        session_folder = f"{expanded_output_folder}/video_regen_{int(time.time())}"
        
        # Prepare input data
        input_data = {
            "original_video": video_path,
            "user_preferences": user_interests,
            "extraction_mode": extraction_mode,
            "screenshot_interval": screenshot_interval,
            "keyframe_threshold": keyframe_threshold,
            "min_interval_frames": min_interval_frames,
            "output_dir": f"{session_folder}/video_analysis",
            "output_folder": session_folder,
            "aspect_ratio": aspect_ratio,
            "output_format": output_format
        }
        
        # Run pipeline
        result = self.run(input_data)
        
        return result
    
    def _prepare_image_generation(
        self, tool_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare input for image generation step.
        
        Takes the list of image prompts and prepares them for batch
        image generation.
        """
        image_prompts = tool_input.get("prompt", [])
        output_folder = tool_input.get("output_folder", "~/Downloads")
        aspect_ratio = tool_input.get("aspect_ratio", "1:1")
        # Always use png for images, regardless of the global output_format
        output_format = "png"
        
        # Create a subfolder for generated images
        image_output_folder = f"{output_folder}/generated_images"
        
        # Display prompts being used
        print(f"\n🎨 IMAGE GENERATION STEP")
        print(f"📁 Output folder: {image_output_folder}")
        print(f"📝 Processing {len(image_prompts)} image prompts:")
        
        for i, prompt in enumerate(image_prompts):
            print(f"   Scene {i+1}: {prompt}")
        
        # Debug: Print the image prompts being processed
        if self.debug:
            print("\n🔍 DEBUG - Image Generation Input:")
            print(f"   Number of prompts: {len(image_prompts)}")
            for i, prompt in enumerate(image_prompts):
                print(f"   Prompt {i+1}: {prompt[:100]}...")
            print(f"   Output folder: {image_output_folder}")
        
        return {
            "prompt": image_prompts,
            "output_folder": image_output_folder,
            "aspect_ratio": aspect_ratio,
            "output_format": output_format
        }
    
    def _extract_image_paths(
        self, tool_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract image paths from image generation output.
        
        The image generation tool returns a list of image paths.
        """
        generated_images = tool_output.get("generated_image_paths", [])
        generation_info = tool_output.get("image_generation_info", {})
        
        # Filter out empty paths and check for errors
        valid_images = []
        errors = []
        
        for i, image_path in enumerate(generated_images):
            if image_path and os.path.exists(image_path):
                valid_images.append(image_path)
                print(f"✅ Generated image {i+1}: {os.path.basename(image_path)}")
            else:
                error_msg = f"Image {i+1} generation failed"
                if isinstance(generation_info, list) and i < len(generation_info):
                    error_info = generation_info[i]
                    if isinstance(error_info, dict) and "error" in error_info:
                        error_msg += f": {error_info['error']}"
                errors.append(error_msg)
                print(f"❌ {error_msg}")
        
        # If no images were generated successfully, create fallback images
        if not valid_images:
            print("\n⚠️  No images generated successfully. Creating fallback images...")
            valid_images = self._create_fallback_images(
                tool_output.get("prompt", []),
                tool_output.get("output_folder", "~/Downloads")
            )
        
        # Debug: Print the generated image paths
        if self.debug:
            print("\n🔍 DEBUG - Image Generation Output:")
            print(f"   Number of valid images: {len(valid_images)}")
            print(f"   Number of errors: {len(errors)}")
            for i, image_path in enumerate(valid_images):
                print(f"   Image {i+1}: {image_path}")
        
        return {
            "generated_image_paths": valid_images,
            "image_generation_info": generation_info,
            "image_generation_errors": errors
        }
    
    def _create_fallback_images(
        self, prompts: List[str], output_folder: str
    ) -> List[str]:
        """
        Create fallback images when image generation fails.
        
        Args:
            prompts: List of prompts that failed to generate images
            output_folder: Output folder for fallback images
            
        Returns:
            List of paths to fallback images
        """
        fallback_images = []
        
        for i, prompt in enumerate(prompts):
            # Create a simple text-based fallback image
            try:
                from PIL import Image, ImageDraw, ImageFont

                # Create a simple image with the prompt text
                img = Image.new('RGB', (512, 512), color='#2c3e50')
                draw = ImageDraw.Draw(img)
                
                # Try to use a default font, fallback to default if not available
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
                except OSError:
                    font = ImageFont.load_default()
                
                # Draw the prompt text
                text = f"Fallback Image {i+1}\n{prompt[:100]}..."
                draw.text((50, 50), text, fill='white', font=font)
                
                # Save the fallback image
                fallback_path = f"{output_folder}/fallback_image_{i+1}.png"
                os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
                img.save(fallback_path)
                
                fallback_images.append(fallback_path)
                print(f"📝 Created fallback image {i+1}: {os.path.basename(fallback_path)}")
                
            except ImportError:
                # If PIL is not available, create a simple text file
                fallback_path = f"{output_folder}/fallback_image_{i+1}.txt"
                os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
                
                with open(fallback_path, 'w') as f:
                    f.write(f"Fallback Image {i+1}\n")
                    f.write(f"Prompt: {prompt}\n")
                    f.write("This is a fallback file because image generation failed.\n")
                
                fallback_images.append(fallback_path)
                print(f"📝 Created fallback file {i+1}: {os.path.basename(fallback_path)}")
        
        return fallback_images
    
    def _prepare_video_generation(
        self, tool_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare input for video generation step.
        
        Takes the list of generated images and video prompts and prepares
        them for batch video generation.
        """
        generated_images = tool_input.get("image", [])
        video_prompts = tool_input.get("prompt", [])
        output_folder = tool_input.get("output_folder", "~/Downloads")
        output_format = tool_input.get("output_format", "mp4")
        
        # Create a subfolder for generated videos
        video_output_folder = f"{output_folder}/generated_videos"
        
        # Display video generation info
        print(f"\n🎬 VIDEO GENERATION STEP")
        print(f"📁 Output folder: {video_output_folder}")
        print(f"🖼️  Processing {len(generated_images)} images:")
        
        for i, (image_path, prompt) in enumerate(
            zip(generated_images, video_prompts)
        ):
            image_name = os.path.basename(image_path) if image_path else "No image"
            print(f"   Video {i+1}: {image_name}")
            print(f"   Prompt: {prompt}")
        
        # Debug: Print the video generation input
        if self.debug:
            print("\n🔍 DEBUG - Video Generation Input:")
            print(f"   Number of images: {len(generated_images)}")
            print(f"   Number of prompts: {len(video_prompts)}")
            for i, (image_path, prompt) in enumerate(
                zip(generated_images, video_prompts)
            ):
                print(f"   Pair {i+1}:")
                print(f"     Image: {image_path}")
                print(f"     Prompt: {prompt[:100]}...")
            print(f"   Output folder: {video_output_folder}")
        
        return {
            "image": generated_images,
            "prompt": video_prompts,
            "output_folder": video_output_folder,
            "output_format": output_format,
            "aspect_ratio": tool_input.get("aspect_ratio", "9:16")
        }
    
    def _extract_video_paths(
        self, tool_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract video paths from video generation output.
        
        The video generation tool returns a list of video paths.
        """
        generated_videos = tool_output.get("generated_video_paths", [])
        generation_info = tool_output.get("video_generation_info", {})
        
        # Filter out empty paths and check for errors
        valid_videos = []
        errors = []
        
        for i, video_path in enumerate(generated_videos):
            if video_path and os.path.exists(video_path):
                valid_videos.append(video_path)
                print(f"✅ Generated video {i+1}: {os.path.basename(video_path)}")
            else:
                error_msg = f"Video {i+1} generation failed"
                if isinstance(generation_info, list) and i < len(generation_info):
                    error_info = generation_info[i]
                    if isinstance(error_info, dict) and "error" in error_info:
                        error_msg += f": {error_info['error']}"
                errors.append(error_msg)
                print(f"❌ {error_msg}")
        
        # Debug: Print the generated video paths
        if self.debug:
            print("\n🔍 DEBUG - Video Generation Output:")
            print(f"   Number of videos: {len(valid_videos)}")
            print(f"   Number of errors: {len(errors)}")
            for i, video_path in enumerate(valid_videos):
                print(f"   Video {i+1}: {video_path}")
        
        return {
            "generated_video_paths": valid_videos,
            "video_generation_info": generation_info,
            "video_generation_errors": errors
        }

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the pipeline with the given input.
        
        Args:
            input_data: Initial input data for the pipeline
            
        Returns:
            Final output from the last step
        """
        # Expand paths
        expanded_video_path = expand_path(input_data["original_video"])
        expanded_output_folder = expand_path(input_data["output_folder"])
        
        # Create output directory if it doesn't exist
        Path(expanded_output_folder).mkdir(parents=True, exist_ok=True)
        
        # Update input data with expanded paths
        input_data["original_video"] = expanded_video_path
        input_data["output_folder"] = expanded_output_folder
        
        # Debug: Show initial input
        if self.debug:
            print("\n🔍 DEBUG - Initial Pipeline Input:")
            for key, value in input_data.items():
                if isinstance(value, list) and len(value) > 3:
                    print(f"   {key}: {len(value)} items")
                    for j, item in enumerate(value[:3]):
                        print(f"     {j+1}: {str(item)[:100]}...")
                    if len(value) > 3:
                        print(f"     ... and {len(value) - 3} more")
                else:
                    print(f"   {key}: {value}")
        
        # Run the parent pipeline with custom execution
        self.logger.info(
            f"Starting pipeline execution with {len(self.steps)} steps"
        )
        
        current_input = input_data.copy()
        
        for i, step in enumerate(self.steps):
            self.logger.info(
                f"Executing step {i+1}/{len(self.steps)}: {step.name}"
            )
            
            # Debug: Show input to this step
            if self.debug:
                print(f"\n🔍 DEBUG - Input to step {i+1} ({step.name}):")
                for key, value in current_input.items():
                    if isinstance(value, list) and len(value) > 3:
                        print(f"   {key}: {len(value)} items")
                        for j, item in enumerate(value[:3]):
                            print(f"     {j+1}: {str(item)[:100]}...")
                        if len(value) > 3:
                            print(f"     ... and {len(value) - 3} more")
                    else:
                        print(f"   {key}: {value}")
            
            # Execute the step
            executor = PipelineStepExecutor(step)
            step_output = executor.execute(current_input)
            
            # Debug: Show output from this step
            if self.debug:
                print(f"\n🔍 DEBUG - Output from step {i+1} ({step.name}):")
                for key, value in step_output.items():
                    if isinstance(value, list) and len(value) > 3:
                        print(f"   {key}: {len(value)} items")
                        for j, item in enumerate(value[:3]):
                            print(f"     {j+1}: {str(item)[:100]}...")
                        if len(value) > 3:
                            print(f"     ... and {len(value) - 3} more")
                    else:
                        print(f"   {key}: {value}")
            
            # Merge step output with current input for next step
            current_input.update(step_output)
            
            self.logger.debug(f"Step {step.name} output: {step_output}")
        
        self.logger.info("Pipeline execution completed")
        
        # Debug: Show final result
        if self.debug:
            print("\n🔍 DEBUG - Final Pipeline Result Keys:")
            for key, value in current_input.items():
                if isinstance(value, list):
                    print(f"   {key}: {len(value)} items")
                else:
                    print(f"   {key}: {value}")
        
        return current_input


def setup_logging():
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def check_environment_setup():
    """
    Check if the environment is properly set up for the pipeline.
    
    Returns:
        bool: True if environment is ready, False otherwise
    """
    missing_vars = []
    
    # Check for required API keys
    if not os.getenv("OPENAI_API_KEY"):
        missing_vars.append("OPENAI_API_KEY")
    
    if not os.getenv("REPLICATE_API_TOKEN"):
        missing_vars.append("REPLICATE_API_TOKEN")
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n📝 Please set these variables in your .env file:")
        print("   Copy env.example to .env and fill in your API keys")
        return False
    
    print("✅ Environment setup looks good!")
    return True


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Regenerate a video based on user interests"
    )
    parser.add_argument(
        "--video-path",
        default="./examples/media-gen/integration_tests/test_video.mp4",
        help="Path to the original video file (supports ~ for home directory). "
             "Default: ./examples/media-gen/integration_tests/test_video.mp4"
    )
    parser.add_argument(
        "--user-interests",
        default="Users like cinematic style with dramatic lighting and "
                "professional video quality",
        help="User interests/preferences for regeneration. "
             "Default: cinematic style with dramatic lighting"
    )
    parser.add_argument(
        "--output-folder",
        default="~/Downloads",
        help="Output folder for generated videos (default: ~/Downloads)"
    )
    parser.add_argument(
        "--extraction-mode",
        choices=["interval", "keyframe"],
        default="interval",
        help="Extraction mode: 'interval' for regular intervals or 'keyframe' for scene changes (default: interval)"
    )
    parser.add_argument(
        "--screenshot-interval",
        type=float,
        default=10.0,
        help="Time interval between screenshots in seconds (for interval mode, default: 10.0)"
    )
    parser.add_argument(
        "--keyframe-threshold",
        type=float,
        default=30.0,
        help="Threshold for keyframe detection (for keyframe mode, default: 30.0)"
    )
    parser.add_argument(
        "--min-interval-frames",
        type=int,
        default=30,
        help="Minimum frames between keyframes (for keyframe mode, default: 30)"
    )
    parser.add_argument(
        "--aspect-ratio",
        default="1:1",
        help="Aspect ratio for generated images (default: 1:1)"
    )
    parser.add_argument(
        "--output-format",
        default="mp4",
        help="Output format for generated videos (default: mp4)"
    )
    parser.add_argument(
        "--image-generator",
        choices=["replicate"],
        default="replicate",
        help="Image generation service to use (default: replicate)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output to see prompts used in each step"
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Check environment setup and exit"
    )
    
    args = parser.parse_args()
    
    # Check environment if requested
    if args.check_env:
        check_environment_setup()
        return
    
    # Expand paths
    expanded_video_path = expand_path(args.video_path)
    
    # Validate input video exists
    if not Path(expanded_video_path).exists():
        print(f"❌ Error: Video file not found: {expanded_video_path}")
        print("💡 Make sure the test video exists or provide a different path")
        sys.exit(1)
    
    # Check environment setup
    if not check_environment_setup():
        sys.exit(1)
    
    # Setup logging
    setup_logging()
    
    print("\n🎬 Starting video regeneration pipeline...")
    print(f"📹 Input video: {expanded_video_path}")
    print(f"🎯 User interests: {args.user_interests}")
    print(f"📁 Output folder: {expand_path(args.output_folder)}")
    print(f"🔍 Extraction mode: {args.extraction_mode}")
    if args.extraction_mode == "interval":
        print(f"⏱️  Screenshot interval: {args.screenshot_interval}s")
    else:
        print(f"🎯 Keyframe threshold: {args.keyframe_threshold}")
        print(f"📏 Min interval frames: {args.min_interval_frames}")
    print(f"⚙️  Image generator: {args.image_generator}")
    print(f"🔧 Debug mode: {args.debug}")
    print("⏱️  Estimated time: 5-15 minutes (depends on video length)")
    print()
    
    try:
        # Initialize tools
        video_understanding = VideoUnderstandingTool()
        
        # Initialize image generation tool
        image_gen = ReplicateImageGen()
        
        # Initialize video generation tool
        video_gen = ReplicateVideoGen()
        
        # Create pipeline
        pipeline = VideoRegenerationPipeline(
            video_understanding_tool=video_understanding,
            image_generation_tool=image_gen,
            video_generation_tool=video_gen,
            debug=args.debug
        )
        
        # Regenerate video
        result = pipeline.regenerate(
            video_path=args.video_path,
            user_interests=args.user_interests,
            output_folder=args.output_folder,
            extraction_mode=args.extraction_mode,
            screenshot_interval=args.screenshot_interval,
            keyframe_threshold=args.keyframe_threshold,
            min_interval_frames=args.min_interval_frames,
            aspect_ratio=args.aspect_ratio,
            output_format=args.output_format
        )
        
        # Output results
        print("\n" + "="*60)
        print("🎬 VIDEO REGENERATION COMPLETE")
        print("="*60)
        
        # Get video paths from the result
        generated_videos = result.get('generated_video_paths', [])
        if generated_videos:
            print("📁 Videos stored at:")
            for i, video_path in enumerate(generated_videos):
                print(f"   Video {i+1}: {video_path}")
                # Show relative path if it's in Downloads
                downloads_path = os.path.expanduser("~/Downloads")
                if video_path.startswith(downloads_path):
                    relative_path = os.path.relpath(video_path, downloads_path)
                    print(f"     📂 Relative to Downloads: {relative_path}")
        else:
            print("❌ No videos generated")
        
        # Get image paths from the result
        generated_images = result.get('generated_image_paths', [])
        if generated_images:
            print("\n🖼️  Generated images:")
            for i, image_path in enumerate(generated_images):
                print(f"   Image {i+1}: {image_path}")
        
        # Get video analysis
        video_analysis = result.get('video_metadata', {})
        if video_analysis:
            print("\n🔍 Video Analysis:")
            total_scenes = video_analysis.get('total_scenes', 'Unknown')
            print(f"   Total scenes: {total_scenes}")
            video_duration = video_analysis.get('video_duration', 'Unknown')
            print(f"   Video duration: {video_duration}")
            extraction_mode = video_analysis.get('extraction_mode', 'Unknown')
            print(f"   Extraction mode: {extraction_mode}")
            
            if extraction_mode == "keyframe":
                keyframe_threshold = video_analysis.get('keyframe_threshold', 'Unknown')
                min_interval_frames = video_analysis.get('min_interval_frames', 'Unknown')
                print(f"   Keyframe threshold: {keyframe_threshold}")
                print(f"   Min interval frames: {min_interval_frames}")
            else:
                screenshot_interval = video_analysis.get(
                    'screenshot_interval', 'Unknown'
                )
                print(f"   Screenshot interval: {screenshot_interval}")
        
        # Get scene descriptions
        scene_descriptions = result.get('scene_descriptions', [])
        if scene_descriptions:
            print("\n🎭 Scene Descriptions:")
            for i, description in enumerate(scene_descriptions):
                print(f"   Scene {i+1}: {description}")
        
        # Get generation info
        video_generation_info = result.get('video_generation_info', [])
        if video_generation_info:
            print("\n📊 Video Generation Info:")
            if isinstance(video_generation_info, list):
                for i, info in enumerate(video_generation_info):
                    if isinstance(info, dict):
                        print(f"   Video {i+1}:")
                        for key, value in info.items():
                            if key == 'prompt':
                                print(f"     {key}: {str(value)[:100]}...")
                            else:
                                print(f"     {key}: {value}")
                    else:
                        print(f"   Video {i+1}: {info}")
            elif isinstance(video_generation_info, dict):
                for key, value in video_generation_info.items():
                    print(f"   {key}: {value}")
            else:
                print(f"   {video_generation_info}")
        
        # Show any errors that occurred
        image_errors = result.get('image_generation_errors', [])
        video_errors = result.get('video_generation_errors', [])
        
        if image_errors or video_errors:
            print("\n⚠️  Errors encountered:")
            for error in image_errors:
                print(f"   Image: {error}")
            for error in video_errors:
                print(f"   Video: {error}")
        
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
