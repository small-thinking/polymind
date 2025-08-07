"""
Replicate image generation tool using various models.

This module provides a real implementation of an image generation tool using
Replicate's API with various image generation models. It integrates seamlessly
with the Polymind framework and supports various image generation parameters.
"""

import replicate
from pathlib import Path

from polymind.core.message import Message

from .media_gen_tool_base import ImageGenerationTool


class ReplicateImageGen(ImageGenerationTool):
    """
    Replicate image generation tool using various models.
    
    This tool uses Replicate's API to generate images using various models
    like WAN, Stable Diffusion, and others. It supports various parameters
    including seed, prompt, aspect ratio, and model-specific options.
    
    Requires Replicate API token to be set in environment variables.
    """

    def __init__(self, model: str = "prunaai/wan-2.2-image", **kwargs):
        """
        Initialize the Replicate image generation tool.
        
        Args:
            model (str): Replicate model identifier (default: "prunaai/wan-2.2-image")
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(
            tool_name="replicate_image_generator",
            descriptions=[
                f"Replicate image generation using {model}",
                "Generate high-quality images from text prompts",
                "Supports various models and parameters"
            ],
            **kwargs
        )
        self._model = model

    def run(self, input: dict) -> dict:
        """
        Generate images using Replicate API.
        
        Args:
            input (dict): Input parameters containing:
                - prompt: Text description(s) of the desired image(s) - can be string or list
                - output_folder: Folder path where to save the image(s) (optional, default: "~/Downloads")
                - seed: Random seed for reproducible results (optional)
                - aspect_ratio: Image aspect ratio (optional, default: "4:3")
                - output_format: Output format (optional, default: "jpeg")
                - quality: Image quality (optional, default: 80)
                - model: Replicate model to use (optional, overrides default)
        
        Returns:
            dict: Dictionary containing:
                - generated_image_paths: List of paths to generated image files
                - image_generation_info: List of generation metadata for each image
        """
        # Extract parameters with defaults
        prompt = input.get("prompt", "")
        output_folder = input.get("output_folder", str(Path.home() / "Downloads"))
        seed = input.get("seed")
        aspect_ratio = input.get("aspect_ratio", "9:16")
        output_format = input.get("output_format", "jpeg")
        quality = input.get("quality", 80)
        model = input.get("model", self._model)
        
        # Handle both single prompt and list of prompts
        if isinstance(prompt, str):
            prompts = [prompt]
        elif isinstance(prompt, list):
            prompts = prompt
        else:
            raise ValueError("Prompt must be a string or list of strings")
        
        generated_images = []
        generation_info = []
        
        # Process each prompt
        for i, single_prompt in enumerate(prompts):
            try:
                # Generate dynamic image name with timestamp to avoid duplication
                import os
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = f"replicate_generated_image_{timestamp}_{i+1}"
                image_name = f"{base_name}.{output_format}"
                
                # Ensure unique filename
                counter = 1
                full_path = f"{output_folder.rstrip('/')}/{image_name}"
                while os.path.exists(full_path):
                    image_name = f"{base_name}_{counter}.{output_format}"
                    full_path = f"{output_folder.rstrip('/')}/{image_name}"
                    counter += 1
                
                # Create full path
                image_path = f"{output_folder.rstrip('/')}/{image_name}"
                
                # Prepare input for Replicate
                replicate_input = {
                    "prompt": single_prompt,
                    "aspect_ratio": aspect_ratio,
                    "quality": quality
                }
                
                # Add seed if provided
                if seed is not None:
                    replicate_input["seed"] = seed
                
                # Ensure directory exists
                output_path = Path(image_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Run the model
                output = replicate.run(model, input=replicate_input)
                
                # Handle different output types from Replicate
                if hasattr(output, 'read'):
                    # Output is a FileOutput object
                    with open(image_path, "wb") as file:
                        file.write(output.read())
                    
                    generated_images.append(image_path)
                    generation_info.append({
                        "model": model,
                        "prompt": single_prompt,
                        "seed": seed,
                        "aspect_ratio": aspect_ratio,
                        "format": output_format,
                        "status": "generated successfully",
                        "replicate_url": None
                    })
                    
                elif isinstance(output, list) and len(output) > 0:
                    # Output is a list of URLs
                    image_url = output[0]
                    import requests

                    # Download the image
                    response = requests.get(image_url)
                    response.raise_for_status()
                    
                    # Save the image
                    with open(image_path, "wb") as file:
                        file.write(response.content)
                    
                    generated_images.append(image_path)
                    generation_info.append({
                        "model": model,
                        "prompt": single_prompt,
                        "seed": seed,
                        "aspect_ratio": aspect_ratio,
                        "format": output_format,
                        "status": "generated successfully",
                        "replicate_url": image_url
                    })
                    
                else:
                    raise ValueError(f"Unexpected output format from Replicate: {type(output)}")
                
            except Exception as e:
                # Add empty path and error info for failed generation
                generated_images.append("")
                generation_info.append({
                    "model": model,
                    "prompt": single_prompt,
                    "error": str(e),
                    "status": "generation failed"
                })
        
        return {
            "generated_image_paths": generated_images,
            "image_generation_info": generation_info
        }

    async def _execute(self, input: Message) -> Message:
        """
        Execute the Replicate image generation using the Polymind framework's Message system.
        
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