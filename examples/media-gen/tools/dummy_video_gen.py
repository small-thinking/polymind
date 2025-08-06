"""
Dummy video generation tool for testing and demonstration purposes.

This module provides a mock implementation of a video generation tool that returns
hardcoded paths instead of actually generating videos. This is useful for testing
the media generation framework structure without requiring real video generation APIs.
"""

from polymind.core.message import Message

from .media_gen_tool_base import VideoGenerationTool


class DummyVideoGen(VideoGenerationTool):
    """
    A dummy video generation tool that returns hardcoded video paths.
    
    This tool simulates video generation by returning predefined video file paths
    instead of actually creating videos. It's designed for:
    - Testing the media generation framework structure
    - Development and debugging without API costs
    - Demonstrating the integration pattern with Polymind agents
    
    In a production environment, this would be replaced with actual video
    generation implementations using services like Runway, Pika Labs, etc.
    """

    def __init__(self, **kwargs):
        """Initialize the DummyVideoGen tool."""
        super().__init__(
            tool_name="dummy_video_generator",
            descriptions=[
                "A dummy video generation tool that returns hardcoded video paths for testing",
                "Mock video generator for development and testing without real API calls", 
                "Placeholder video generation tool that simulates video creation workflow"
            ],
            **kwargs
        )

    def run(self, input: dict) -> dict:
        """
        Simulate video generation by returning a hardcoded video path.
        
        Args:
            input (dict): Input parameters containing:
                - prompt: Text description of the desired video
                - num_frames: Number of frames (optional, default: 81)
                - resolution: Video resolution (optional, default: "480p")
                - image: URI of starting image (optional)
        
        Returns:
            dict: Dictionary containing:
                - video_path: Hardcoded path to a dummy video
                - generation_info: Mock generation metadata
        """
        # Extract parameters with defaults
        prompt = input.get("prompt", "")
        num_frames = input.get("num_frames", 81)
        resolution = input.get("resolution", "480p")
        image = input.get("image", None)
        
        # Return hardcoded dummy video path and metadata
        dummy_video_path = "/tmp/dummy_generated_video.mp4"
        
        generation_info = {
            "model": "dummy-video-generator-v1.0",
            "prompt": prompt,
            "num_frames": str(num_frames),
            "resolution": resolution,
            "seed": "67890",
            "status": "generated (dummy)"
        }
        
        if image:
            generation_info["starting_image"] = image
        
        return {
            "video_path": dummy_video_path,
            "generation_info": generation_info
        }

    async def _execute(self, input: Message) -> Message:
        """
        Execute the dummy video generation using the Polymind framework's Message system.
        
        Args:
            input (Message): Input message containing generation parameters
            
        Returns:
            Message: Output message with generated video information
        """
        # Convert Message to dict for the run method
        input_dict = input.content
        
        # Call the run method
        result = self.run(input_dict)
        
        # Return result wrapped in a Message
        return Message(content=result) 