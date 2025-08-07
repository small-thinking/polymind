"""
Abstract base classes for media generation tools in the Polymind framework.

This module defines the foundational interfaces for image and video generation
tools that can be integrated into Polymind agents. These base classes provide
a consistent API for media generation operations and ensure compatibility with
the framework's tool system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

from polymind.core.tool import BaseTool, Param


class ImageGenerationTool(BaseTool, ABC):
    """
    Abstract base class for image generation tools within the Polymind framework.
    
    This class defines the interface that all image generation tools must
    implement. It inherits from BaseTool to integrate seamlessly with
    Polymind's agent system, allowing AI agents to generate images as part
    of their workflow.
    
    Implementations should provide concrete image generation capabilities
    using various APIs or models (e.g., DALL-E, Stable Diffusion,
    Midjourney).
    """

    def input_spec(self) -> List[Param]:
        """Define the input parameters for image generation."""
        return [
            Param(
                name="prompt",
                type="str",
                required=True,
                description="Text prompt describing the image to generate",
                example="A beautiful sunset over a mountain landscape"
            ),
            Param(
                name="aspect_ratio",
                type="str",
                required=False,
                description="Aspect ratio of the generated image (e.g., '4:3', '16:9')",
                example="4:3"
            ),
            Param(
                name="output_format",
                type="str",
                required=False,
                description="Output format for the generated image",
                example="jpg"
            ),
            Param(
                name="output_folder",
                type="str",
                required=False,
                description="Folder path where to save the generated image",
                example="/path/to/save/images/"
            )
        ]

    def output_spec(self) -> List[Param]:
        """Define the output parameters for image generation."""
        return [
            Param(
                name="image_path",
                type="str",
                required=True,
                description="Path to the generated image file",
                example="/path/to/generated_image.jpg"
            ),
            Param(
                name="generation_info",
                type="Dict[str, str]",
                required=False,
                description="Additional information about the generation process",
                example='{"model": "stable-diffusion", "seed": "42"}'
            )
        ]

    @abstractmethod
    def run(self, input: dict) -> dict:
        """
        Generate an image based on the provided input parameters.
        
        Args:
            input (dict): Dictionary containing generation parameters such as:
                - prompt: Text description of the desired image
                - aspect_ratio: Image aspect ratio (optional, default: "4:3")
                - output_format: Output format (optional, default: "jpg")
                - output_folder: Folder path to save the image (optional)
        
        Returns:
            dict: Dictionary containing:
                - image_path: Path to the generated image file
                - generation_info: Additional metadata (optional)
        """
        pass


class VideoGenerationTool(BaseTool, ABC):
    """
    Abstract base class for video generation tools within the Polymind framework.
    
    This class defines the interface that all video generation tools must
    implement. It inherits from BaseTool to integrate seamlessly with
    Polymind's agent system, allowing AI agents to generate videos as part
    of their workflow.
    
    Implementations should provide concrete video generation capabilities
    using various APIs or models (e.g., Runway, Pika Labs, or other video
    generation services).
    """

    def input_spec(self) -> List[Param]:
        """Define the input parameters for video generation."""
        return [
            Param(
                name="prompt",
                type="str", 
                required=True,
                description="Text prompt describing the video to generate",
                example="A timelapse of a flower blooming in spring"
            ),
            Param(
                name="num_frames",
                type="int",
                required=False,
                description="Number of frames for the video",
                example="81"
            ),
            Param(
                name="resolution",
                type="str",
                required=False,
                description="Video resolution (e.g., '480p', '720p', '1080p')",
                example="480p"
            ),
            Param(
                name="image",
                type="str",
                required=False,
                description="URI of an image to use as starting point",
                example="https://example.com/starting_image.jpg"
            ),
            Param(
                name="aspect_ratio",
                type="str",
                required=False,
                description="Aspect ratio of the generated video (e.g., '9:16', '16:9')",
                example="9:16"
            )
        ]

    def output_spec(self) -> List[Param]:
        """Define the output parameters for video generation."""
        return [
            Param(
                name="video_path",
                type="str",
                required=True,
                description="Path to the generated video file",
                example="/path/to/generated_video.mp4"
            ),
            Param(
                name="generation_info",
                type="Dict[str, str]",
                required=False,
                description="Additional information about the generation process",
                example='{"model": "runway-gen2", "frames": "81"}'
            )
        ]

    @abstractmethod
    def run(self, input: dict) -> dict:
        """
        Generate a video based on the provided input parameters.
        
        Args:
            input (dict): Dictionary containing generation parameters such as:
                - prompt: Text description of the desired video
                - num_frames: Number of frames (optional, default: 81)
                - resolution: Video resolution (optional, default: "480p")
                - image: URI of starting image (optional)
                - aspect_ratio: Video aspect ratio (optional, default: "9:16")
        
        Returns:
            dict: Dictionary containing:
                - video_path: Path to the generated video file
                - generation_info: Additional metadata (optional)
        """
        pass 