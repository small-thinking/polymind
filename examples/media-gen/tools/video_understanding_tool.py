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
from utils.video_utils import ScreenshotInfo, extract_key_frames, extract_screenshots

from polymind.core.tool import BaseTool, Param
from polymind.core.utils import encode_image_to_base64


class VideoUnderstandingTool(BaseTool):
    """
    Tool for video understanding using OpenAI's GPT-4o-mini API.
    
    This tool extracts screenshots from videos using either interval-based
    or keyframe-based extraction and generates coherent image generation 
    prompts for each scene. The prompts are designed to work together to 
    create a logical sequence of images that can be used for video generation.
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
        6. Ignore the text in the screenshot, if they are about the brand. Only keep non-branding words.
        
        For each screenshot, provide:
        - A detailed image generation prompt, including the scene, object, lighting, camera (e.g. overhead, close-up, etc.) 
            and image aesthetic style (cartoon, realistic, cyberpunk, etc.)
        - A detailed image to video generation prompt, including the potential action of each objects, 
            and the camera movement (e.g. pan, zoom, etc.)

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
            ]
        )
        
        self.api_key = api_key
        self.model = model
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)

    def input_spec(self) -> List[Param]:
        """
        Define the input specification for the tool.
        
        Returns:
            List[Param]: List of input parameters
        """
        return [
            Param(
                name="video_path",
                type="string",
                description="Path to the video file to analyze",
                required=True
            ),
            Param(
                name="user_preference",
                type="string",
                description="User's preference for image generation style and content",
                required=False,
                default="Create detailed, cinematic image generation prompts"
            ),
            Param(
                name="extraction_mode",
                type="string",
                description="Extraction mode: 'interval' for regular intervals or 'keyframe' for scene changes",
                required=False,
                default="interval"
            ),
            Param(
                name="screenshot_interval",
                type="number",
                description="Time interval between screenshots (for interval mode)",
                required=False,
                default=10.0
            ),
            Param(
                name="keyframe_threshold",
                type="number",
                description="Threshold for keyframe detection (for keyframe mode)",
                required=False,
                default=30.0
            ),
            Param(
                name="min_interval_frames",
                type="number",
                description="Minimum frames between keyframes (for keyframe mode)",
                required=False,
                default=30
            ),
            Param(
                name="output_dir",
                type="string",
                description="Directory to save extracted screenshots",
                required=False,
                default="~/Downloads/video_understanding"
            ),
            Param(
                name="max_tokens",
                type="number",
                description="Maximum tokens in OpenAI response",
                required=False,
                default=2000
            )
        ]

    def output_spec(self) -> List[Param]:
        """
        Define the output specification for the tool.
        
        Returns:
            List[Param]: List of output parameters
        """
        return [
            Param(
                name="image_prompts",
                type="array",
                description="List of image generation prompts for each scene"
            ),
            Param(
                name="video_prompts",
                type="array",
                description="List of video generation prompts for each scene"
            ),
            Param(
                name="scene_descriptions",
                type="array",
                description="List of scene descriptions"
            ),
            Param(
                name="screenshot_paths",
                type="array",
                description="List of paths to extracted screenshots"
            ),
            Param(
                name="metadata",
                type="object",
                description="Additional metadata about the analysis"
            )
        ]

    def _extract_screenshots(
        self, 
        video_path: str, 
        extraction_mode: str = "interval",
        interval: float = 10.0,
        keyframe_threshold: float = 30.0,
        min_interval_frames: int = 30,
        output_dir: str = "~/Downloads/video_understanding"
    ) -> List[ScreenshotInfo]:
        """
        Extract screenshots from the video using video_utils.
        
        Args:
            video_path (str): Path to the video file
            extraction_mode (str): 'interval' or 'keyframe'
            interval (float): Time interval between screenshots (for interval mode)
            keyframe_threshold (float): Threshold for keyframe detection (for keyframe mode)
            min_interval_frames (int): Minimum frames between keyframes (for keyframe mode)
            output_dir (str): Directory to save screenshots
            
        Returns:
            List[ScreenshotInfo]: List of screenshot information
        """
        try:
            if extraction_mode.lower() == "keyframe":
                # Use keyframe extraction
                key_frames = extract_key_frames(
                    video_path=video_path,
                    threshold=keyframe_threshold,
                    min_interval_frames=min_interval_frames,
                    output_dir=output_dir,
                    filename_prefix="video_scene"
                )
                
                # Convert KeyFrameInfo to ScreenshotInfo for compatibility
                screenshots = []
                for key_frame in key_frames:
                    screenshot = ScreenshotInfo(
                        frame_number=key_frame.frame_number,
                        timestamp=key_frame.timestamp,
                        timestamp_str=key_frame.timestamp_str,
                        file_path=key_frame.file_path
                    )
                    screenshots.append(screenshot)
                
                print(f"🔍 Extracted {len(screenshots)} key frames using scene change detection")
                return screenshots
            else:
                # Use interval-based extraction (default)
                screenshots = extract_screenshots(
                    video_path=video_path,
                    interval_seconds=interval,
                    output_dir=output_dir,
                    filename_prefix="video_scene"
                )
                
                print(f"📸 Extracted {len(screenshots)} screenshots at {interval}s intervals")
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
                - extraction_mode: 'interval' or 'keyframe' (optional, default: 'interval')
                - screenshot_interval: Time interval between screenshots (for interval mode, optional, default: 10.0)
                - keyframe_threshold: Threshold for keyframe detection (for keyframe mode, optional, default: 30.0)
                - min_interval_frames: Minimum frames between keyframes (for keyframe mode, optional, default: 30)
                - output_dir: Directory to save screenshots (optional, default: ~/Downloads/video_understanding)
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
        extraction_mode = input.get("extraction_mode", "interval")
        screenshot_interval = input.get("screenshot_interval", 10.0)
        keyframe_threshold = input.get("keyframe_threshold", 30.0)
        min_interval_frames = input.get("min_interval_frames", 30)
        output_dir = input.get("output_dir", "~/Downloads/video_understanding")
        max_tokens = input.get("max_tokens", 2000)
        
        if not video_path:
            raise ValueError("video_path is required")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Extract screenshots from video
        screenshots = self._extract_screenshots(
            video_path=video_path,
            extraction_mode=extraction_mode,
            interval=screenshot_interval,
            keyframe_threshold=keyframe_threshold,
            min_interval_frames=min_interval_frames,
            output_dir=output_dir
        )
        
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
            "extraction_mode": extraction_mode,
            "total_scenes": str(len(screenshots)),
            "video_duration": f"{screenshots[-1].timestamp:.1f}s" if screenshots else "0s",
            "output_directory": output_dir
        }
        
        # Add mode-specific metadata
        if extraction_mode.lower() == "keyframe":
            metadata.update({
                "keyframe_threshold": str(keyframe_threshold),
                "min_interval_frames": str(min_interval_frames)
            })
        else:
            metadata.update({
                "screenshot_interval": f"{screenshot_interval}s"
            })
        
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
