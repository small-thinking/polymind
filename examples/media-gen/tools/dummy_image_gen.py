"""
Dummy image generation tool for testing and demonstration purposes.

This module provides a mock implementation of an image generation tool that returns
hardcoded paths instead of actually generating images. This is useful for testing
the media generation framework structure without requiring real image generation APIs.
"""

from polymind.core.message import Message

from .media_gen_tool_base import ImageGenerationTool


class DummyImageGen(ImageGenerationTool):
    """
    A dummy image generation tool that returns hardcoded image paths.
    
    This tool simulates image generation by returning predefined image file paths
    instead of actually creating images. It's designed for:
    - Testing the media generation framework structure
    - Development and debugging without API costs
    - Demonstrating the integration pattern with Polymind agents
    
    In a production environment, this would be replaced with actual image
    generation implementations using services like DALL-E, Stable Diffusion, etc.
    """

    def __init__(self, **kwargs):
        """Initialize the DummyImageGen tool."""
        super().__init__(
            tool_name="dummy_image_generator",
            descriptions=[
                "A dummy image generation tool that returns hardcoded image paths for testing",
                "Mock image generator for development and testing without real API calls",
                "Placeholder image generation tool that simulates image creation workflow"
            ],
            **kwargs
        )

    def run(self, input: dict) -> dict:
        """
        Simulate image generation by returning a hardcoded image path.
        
        Args:
            input (dict): Input parameters containing:
                - prompt: Text description of the desired image
                - aspect_ratio: Image aspect ratio (optional, default: "4:3")
                - output_format: Output format (optional, default: "jpg")
        
        Returns:
            dict: Dictionary containing:
                - image_path: Hardcoded path to a dummy image
                - generation_info: Mock generation metadata
        """
        # Extract parameters with defaults
        prompt = input.get("prompt", "")
        aspect_ratio = input.get("aspect_ratio", "4:3")
        output_format = input.get("output_format", "jpg")
        
        # Return hardcoded dummy image path and metadata
        dummy_image_path = f"/tmp/dummy_generated_image.{output_format}"
        
        return {
            "image_path": dummy_image_path,
            "generation_info": {
                "model": "dummy-generator-v1.0",
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "output_format": output_format,
                "seed": "12345",
                "status": "generated (dummy)",
                "note": "This is a dummy implementation. Real tools would use API keys from .env"
            }
        }

    async def _execute(self, input: Message) -> Message:
        """
        Execute the dummy image generation using the Polymind framework's Message system.
        
        Args:
            input (Message): Input message containing generation parameters
            
        Returns:
            Message: Output message with generated image information
        """
        # Convert Message to dict for the run method
        input_dict = input.content
        
        # Call the run method
        result = self.run(input_dict)
        
        # Return result wrapped in a Message
        return Message(content=result) 