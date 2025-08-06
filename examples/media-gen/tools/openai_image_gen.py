"""
OpenAI image generation tool using gpt-4o-mini.

This module provides a real implementation of an image generation tool using
OpenAI's gpt-4o-mini model with image generation capabilities. It integrates
seamlessly with the Polymind framework and supports various image generation
parameters.
"""

import base64

from openai import OpenAI
from pathlib import Path

from polymind.core.message import Message

from .media_gen_tool_base import ImageGenerationTool


class OpenAIImageGen(ImageGenerationTool):
    """
    OpenAI image generation tool using gpt-4o-mini.
    
    This tool uses OpenAI's gpt-4o-mini model with image generation capabilities
    to create images based on text prompts. It supports various parameters
    including size, quality, format, compression, and background options.
    
    Requires OpenAI API key to be set in environment variables.
    """

    def __init__(self, **kwargs):
        """Initialize the OpenAI image generation tool."""
        super().__init__(
            tool_name="openai_image_generator",
            descriptions=[
                "OpenAI image generation using gpt-4o-mini model",
                "Generate high-quality images from text prompts",
                "Supports various image parameters (size, quality, format)"
            ],
            **kwargs
        )

    def run(self, input: dict) -> dict:
        """
        Generate an image using OpenAI's gpt-4o-mini model.
        
        Args:
            input (dict): Input parameters containing:
                - prompt: Text description of the desired image
                - aspect_ratio: Image aspect ratio (optional, default: "1:1")
                - output_format: Output format (optional, default: "png")
                - image_path: Path to save the image (optional)
                - size: Image dimensions (optional, default: "1024x1024")
                - quality: Rendering quality (optional, default: "low")
                - compression: Compression level 0-100% (optional, default: 80)
                - background: Transparent or opaque (optional, default: "opaque")
        
        Returns:
            dict: Dictionary containing:
                - image_path: Path to the generated image file
                - generation_info: Generation metadata
        """
        # Extract parameters with defaults
        prompt = input.get("prompt", "")
        output_format = input.get("output_format", "png")
        output_folder = input.get("output_folder", str(Path.home() / "Downloads"))
        size = input.get("size", "1024x1024")
        quality = input.get("quality", "low")
        compression = input.get("compression", 80)
        background = input.get("background", "opaque")
        
        # Generate dynamic image name with timestamp to avoid duplication
        import os
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"openai_generated_image_{timestamp}"
        image_name = f"{base_name}.{output_format}"
        
        # Ensure unique filename
        counter = 1
        while os.path.exists(f"{output_folder.rstrip('/')}/{image_name}"):
            image_name = f"{base_name}_{counter}.{output_format}"
            counter += 1
        
        # Create full path
        image_path = f"{output_folder.rstrip('/')}/{image_name}"
        
        # Generate image using OpenAI API
        try:
            client = OpenAI()
            response = client.responses.create(
                model="gpt-4o-mini",
                input=prompt,
                tools=[{
                    "type": "image_generation"
                }]
            )
            
            # Extract image data from response
            image_data = [
                output.result
                for output in response.output
                if output.type == "image_generation_call"
            ]
            
            if not image_data:
                raise RuntimeError("No image data received from OpenAI API")
            
            # Decode base64 image data
            image_base64 = image_data[0]
            image_bytes = base64.b64decode(image_base64)
            
            # Ensure directory exists
            output_path = Path(image_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save image to file
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            
            return {
                "image_path": image_path,
                "generation_info": {
                    "model": "gpt-4o-mini",
                    "prompt": prompt,
                    "size": size,
                    "quality": quality,
                    "format": output_format,
                    "compression": compression,
                    "background": background,
                    "status": "generated successfully"
                }
            }
            
        except Exception as e:
            return {
                "image_path": "",
                "generation_info": {
                    "model": "gpt-4o-mini",
                    "prompt": prompt,
                    "error": str(e),
                    "status": "generation failed"
                }
            }

    async def _execute(self, input: Message) -> Message:
        """
        Execute the OpenAI image generation using the Polymind framework's Message system.
        
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