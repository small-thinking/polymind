"""
Video understanding tool using OpenAI's GPT-4o-mini API.

This module provides a tool for analyzing videos by extracting screenshots
and generating coherent image generation prompts for each scene. It leverages
video_utils.py for screenshot extraction and image understanding capabilities
for analysis.
"""

import json
import os
from typing import Any, ClassVar, Dict, List, Optional

from openai import OpenAI
from utils.video_utils import ScreenshotInfo, extract_screenshots

from polymind.core.tool import BaseTool, Param
from polymind.core.utils import encode_image_to_base64


class VideoUnderstandingTool(BaseTool):
    """
    Tool for video understanding using OpenAI's GPT-4o-mini API.
    
    This tool extracts screenshots from videos at regular intervals and
    generates coherent image generation prompts for each scene. The prompts
    are designed to work together to create a logical sequence of images
    that can be used for video generation.
    """

    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    client: Optional[Any] = None

    prompt: ClassVar[str] = """
        Analyze these video screenshots and generate image generation prompts 
        for each scene.
        
        User preference: {user_preference}
        
        Requirements:
        1. Each prompt should be detailed and specific enough for image 
           generation
        2. Prompts should maintain logical coherence between scenes
        3. Consider the visual flow and narrative progression
        4. Include relevant details like lighting, composition, mood, and 
           style
        5. Ensure prompts work together to tell a coherent visual story
        
        For each screenshot, provide:
        - A detailed image generation prompt
        - A detailed image to video generation prompt

        Both should be <100 words.
        
        Respond in JSON format with the following structure:
        {{
            "scenes": [
                {{
                    "t2i_prompt": "detailed image generation prompt",
                    "i2v_prompt": "detailed video generation prompt"
                }}
            ]
        }}
    """

    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: str = "gpt-4o-mini", 
        **kwargs
    ):
        """
        Initialize the video understanding tool.
        
        Args:
            api_key (Optional[str]): OpenAI API key. If None, will use 
                environment variable.
            model (str): OpenAI model to use for video understanding
        """
        # Set the API key
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment "
                "variable or pass api_key parameter."
            )
        
        # Initialize the parent class with the required fields
        super().__init__(
            tool_name="video_understanding",
            descriptions=[
                "Analyze videos by extracting screenshots and generating "
                "image prompts",
                "Video understanding and scene analysis tool",
                "AI tool for generating coherent image prompts from video "
                "scenes"
            ],
            api_key=api_key,
            model=model,
            **kwargs
        )
        
        # Initialize the OpenAI client
        self.client = OpenAI(api_key=self.api_key)

    def input_spec(self) -> List[Param]:
        """Define the input parameters for video understanding."""
        return [
            Param(
                name="video_path",
                type="str",
                required=True,
                description="Path to the video file to analyze",
                example="/path/to/video.mp4"
            ),
            Param(
                name="user_preference",
                type="str",
                required=False,
                description=(
                    "User's preference for the generated image prompts "
                    "(style, theme, etc.)"
                ),
                example="Create prompts in a cinematic style with dramatic lighting"
            ),
            Param(
                name="screenshot_interval",
                type="float",
                required=False,
                description="Time interval between screenshots in seconds "
                           "(default: 10.0)",
                example="10.0"
            ),
            Param(
                name="output_dir",
                type="str",
                required=False,
                description="Directory to save extracted screenshots (default: ~/Downloads)",
                example="~/Downloads/video_screenshots"
            ),
            Param(
                name="max_tokens",
                type="int",
                required=False,
                description="Maximum number of tokens in the response",
                example="2000"
            ),
        ]

    def output_spec(self) -> List[Param]:
        """Define the output parameters for video understanding."""
        return [
            Param(
                name="image_prompts",
                type="List[str]",
                required=True,
                description="List of image generation prompts for each scene",
                example='["A cat sitting on a windowsill in golden hour light"]'
            ),
            Param(
                name="video_prompts",
                type="List[str]",
                required=True,
                description="List of video generation prompts for each scene",
                example='["The cat jumps down and walks across the room"]'
            ),
            Param(
                name="scene_descriptions",
                type="List[str]",
                required=True,
                description="Descriptions of each scene extracted from the video",
                example='["Scene 1: Cat on windowsill"]'
            ),
            Param(
                name="screenshot_paths",
                type="List[str]",
                required=True,
                description="Paths to the extracted screenshot files",
                example='["/path/to/screenshot_1.jpg"]'
            ),
            Param(
                name="metadata",
                type="Dict[str, str]",
                required=False,
                description="Additional metadata about the analysis",
                example='{"model": "gpt-4o-mini", "total_scenes": "5"}'
            )
        ]

    def _extract_screenshots(self, video_path: str, interval: float, output_dir: str) -> List[ScreenshotInfo]:
        """
        Extract screenshots from the video using video_utils.
        
        Args:
            video_path (str): Path to the video file
            interval (float): Time interval between screenshots
            output_dir (str): Directory to save screenshots
            
        Returns:
            List[ScreenshotInfo]: List of screenshot information
        """
        try:
            screenshots = extract_screenshots(
                video_path=video_path,
                interval_seconds=interval,
                output_dir=output_dir,
                filename_prefix="video_scene"
            )
            return screenshots
        except Exception as e:
            raise RuntimeError(f"Failed to extract screenshots from video: {e}")

    def _analyze_screenshots(self, screenshots: List[ScreenshotInfo], user_preference: str) -> List[Dict[str, Any]]:
        """
        Analyze screenshots and generate image prompts using OpenAI.
        
        Args:
            screenshots (List[ScreenshotInfo]): List of screenshot information
            user_preference (str): User's preference for image generation
            
        Returns:
            List[Dict[str, Any]]: List of analysis results for each screenshot
        """
        if not screenshots:
            return []
        
        # Prepare the analysis prompt
        base_prompt = self.prompt.format(user_preference=user_preference)
        
        # Prepare content for OpenAI API
        content = [{"type": "text", "text": base_prompt}]
        
        # Add screenshots to content
        for screenshot in screenshots:
            try:
                base64_image = encode_image_to_base64(screenshot.file_path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
            except Exception as e:
                print(f"Warning: Failed to process screenshot {screenshot.file_path}: {e}")
                continue
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            analysis = response.choices[0].message.content
            
            # Parse JSON response
            try:
                analysis_dict = json.loads(analysis)
                return analysis_dict.get("scenes", [])
            except json.JSONDecodeError:
                raise RuntimeError("Failed to parse OpenAI response as JSON")
                
        except Exception as e:
            raise RuntimeError(f"Failed to analyze screenshots: {e}")

    def run(self, input: dict) -> dict:
        """
        Analyze video and generate coherent image generation prompts.
        
        Args:
            input (dict): Dictionary containing:
                - video_path: Path to the video file
                - user_preference: User's preference for image generation (optional)
                - screenshot_interval: Time interval between screenshots (optional, default: 10.0)
                - output_dir: Directory to save screenshots (optional, default: ~/Downloads)
                - max_tokens: Maximum tokens in response (optional, default: 2000)
        
        Returns:
            dict: Dictionary containing:
                - image_prompts: List of image generation prompts
                - video_prompts: List of video generation prompts
                - scene_descriptions: List of scene descriptions
                - screenshot_paths: List of screenshot file paths
                - metadata: Additional metadata
        """
        # Extract parameters
        video_path = input.get("video_path")
        user_preference = input.get("user_preference", "Create detailed, cinematic image generation prompts")
        screenshot_interval = input.get("screenshot_interval", 10.0)
        output_dir = input.get("output_dir", "~/Downloads/video_understanding")
        max_tokens = input.get("max_tokens", 2000)
        
        if not video_path:
            raise ValueError("video_path is required")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Extract screenshots from video
        screenshots = self._extract_screenshots(video_path, screenshot_interval, output_dir)
        
        if not screenshots:
            raise RuntimeError("No screenshots were extracted from the video")
        
        # Analyze screenshots and generate prompts
        scene_analyses = self._analyze_screenshots(screenshots, user_preference)
        
        # Extract results
        image_prompts = []
        video_prompts = []
        scene_descriptions = []
        screenshot_paths = [s.file_path for s in screenshots]
        
        for i, analysis in enumerate(scene_analyses):
            if i < len(screenshots):
                # Use analysis if available, otherwise create basic prompts
                if isinstance(analysis, dict):
                    image_prompts.append(
                        analysis.get("t2i_prompt", f"Scene {i+1} from video")
                    )
                    video_prompts.append(
                        analysis.get("i2v_prompt", f"Scene {i+1} action")
                    )
                    scene_descriptions.append(f"Scene {i+1}")
                else:
                    image_prompts.append(f"Scene {i+1} from video")
                    video_prompts.append(f"Scene {i+1} action")
                    scene_descriptions.append(f"Scene {i+1}")
        
        # Prepare metadata
        metadata = {
            "model": self.model,
            "total_scenes": str(len(screenshots)),
            "video_duration": f"{screenshots[-1].timestamp:.1f}s" if screenshots else "0s",
            "screenshot_interval": f"{screenshot_interval}s",
            "output_directory": output_dir
        }
        
        return {
            "image_prompts": image_prompts,
            "video_prompts": video_prompts,
            "scene_descriptions": scene_descriptions,
            "screenshot_paths": screenshot_paths,
            "metadata": metadata
        }

    @property
    def tool_name(self) -> str:
        """Return the name of the tool."""
        return "video_understanding"

    @property
    def tool_description(self) -> str:
        """Return the description of the tool."""
        return "Analyze videos by extracting screenshots and generating coherent image generation prompts for each scene"
