"""
Replicate video generation tool using WAN 2.2 i2v fast model.

This module provides a real implementation of a video generation tool using
Replicate's API with the WAN 2.2 i2v fast model. It integrates seamlessly
with the Polymind framework and supports image-to-video generation with
text prompts.
"""

import base64
import os
from typing import Union

import replicate
import requests
from pathlib import Path

from polymind.core.message import Message

from .media_gen_tool_base import VideoGenerationTool


class ReplicateVideoGen(VideoGenerationTool):
    """
    Replicate video generation tool using WAN 2.2 i2v fast model.
    
    This tool uses Replicate's API to generate videos from images and text
    prompts using the WAN 2.2 i2v fast model. It supports various parameters
    including image input (file path, URL, or data URI) and text prompts.
    
    Requires Replicate API token to be set in environment variables.
    """

    def __init__(self, model: str = "wan-video/wan-2.2-i2v-fast", **kwargs):
        """
        Initialize the Replicate video generation tool.
        
        Args:
            model (str): Replicate model identifier 
                        (default: "wan-video/wan-2.2-i2v-fast")
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(
            tool_name="replicate_video_generator",
            descriptions=[
                f"Replicate video generation using {model}",
                "Generate videos from images and text prompts",
                "Supports image-to-video generation with WAN 2.2 i2v fast "
                "model"
            ],
            **kwargs
        )
        self._model = model

    def _prepare_image_input(self, image_input: Union[str, Path]) -> str:
        """
        Prepare image input for Replicate API.
        
        Args:
            image_input: Image path, URL, or data URI
            
        Returns:
            str: Prepared image input for Replicate API
        """
        image_str = str(image_input)
        
        # If it's already a data URI or URL, return as is
        if image_str.startswith(('data:', 'http://', 'https://')):
            return image_str
        
        # If it's a file path, convert to data URI for Replicate API
        image_path = Path(image_str)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Always convert to data URI for Replicate API
        with open(image_path, 'rb') as file:
            data = base64.b64encode(file.read()).decode('utf-8')
            return f"data:application/octet-stream;base64,{data}"

    def run(self, input: dict) -> dict:
        """
        Generate videos using Replicate WAN 2.2 i2v fast API with progress monitoring.
        
        Args:
            input (dict): Input parameters containing:
                - image: Image path(s), URL(s), or data URI(s) - can be string or list (required)
                - prompt: Text description(s) of the desired video(s) - can be string or list (required)
                - output_folder: Folder path where to save the video(s) 
                  (optional, default: "~/Downloads")
                - output_format: Output format (optional, default: "mp4")
                - model: Replicate model to use (optional, overrides default)
                - timeout: Timeout in seconds (optional, default: 300)
                - progress_interval: Progress update interval in seconds (optional, default: 5)
        
        Returns:
            dict: Dictionary containing:
                - generated_video_paths: List of paths to generated video files
                - video_generation_info: List of generation metadata for each video
        """
        # Extract parameters with defaults
        image_input = input.get("image", "")
        prompt = input.get("prompt", "")
        output_folder = input.get(
            "output_folder", str(Path.home() / "Downloads")
        )
        output_format = input.get("output_format", "mp4")
        aspect_ratio = input.get("aspect_ratio", "9:16")
        model = input.get("model", self._model)
        timeout = input.get("timeout", 300)  # 5 minutes default
        progress_interval = input.get("progress_interval", 5)  # 5 seconds default
        
        # Handle both single and multiple inputs
        if isinstance(image_input, str):
            images = [image_input]
        elif isinstance(image_input, list):
            images = image_input
        else:
            raise ValueError("Image input must be a string or list of strings")
        
        if isinstance(prompt, str):
            prompts = [prompt]
        elif isinstance(prompt, list):
            prompts = prompt
        else:
            raise ValueError("Prompt must be a string or list of strings")
        
        # Ensure we have matching numbers of images and prompts
        if len(images) != len(prompts):
            raise ValueError(
                f"Number of images ({len(images)}) must match number of "
                f"prompts ({len(prompts)})"
            )
        
        generated_videos = []
        generation_info = []
        
        # Process each image-prompt pair
        for i, (single_image, single_prompt) in enumerate(zip(images, prompts)):
            try:
                if not single_image:
                    raise ValueError("Image input is required")
                
                if not single_prompt:
                    raise ValueError("Text prompt is required")
                
                # Generate dynamic video name with timestamp to avoid duplication
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = f"replicate_generated_video_{timestamp}_{i+1}"
                video_name = f"{base_name}.{output_format}"
                
                # Ensure unique filename
                counter = 1
                full_path = f"{output_folder.rstrip('/')}/{video_name}"
                while os.path.exists(full_path):
                    video_name = f"{base_name}_{counter}.{output_format}"
                    full_path = f"{output_folder.rstrip('/')}/{video_name}"
                    counter += 1
                
                # Create full path
                video_path = f"{output_folder.rstrip('/')}/{video_name}"
                
                # Prepare image input
                prepared_image = self._prepare_image_input(single_image)
                
                # Prepare input for Replicate
                replicate_input = {
                    "image": prepared_image,
                    "prompt": single_prompt,
                    "aspect_ratio": aspect_ratio
                }
                
                # Ensure directory exists
                output_path = Path(video_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create prediction in background
                import time
                start_time = time.time()
                
                # Create prediction using the model string directly
                prediction = replicate.predictions.create(
                    model=model,
                    input=replicate_input
                )
                
                print(f"🔄 Started video generation {i+1} (ID: {prediction.id})")
                
                # Monitor progress with timeout
                last_progress_time = start_time
                while True:
                    # Check timeout
                    if time.time() - start_time > timeout:
                        prediction.cancel()
                        raise TimeoutError(
                            f"Video generation {i+1} timed out after {timeout} "
                            "seconds"
                        )
                    
                    # Reload prediction to get latest status
                    prediction.reload()
                    
                    # Print progress updates
                    if time.time() - last_progress_time >= progress_interval:
                        elapsed = int(time.time() - start_time)
                        print(f"⏱️  Video {i+1} Status: {prediction.status} "
                              f"(elapsed: {elapsed}s)")
                        if prediction.logs:
                            print(f"📝 Logs: {prediction.logs[-200:]}...")
                        last_progress_time = time.time()
                    
                    # Check if completed
                    if prediction.status == "succeeded":
                        print(f"✅ Video generation {i+1} completed!")
                        break
                    elif prediction.status == "failed":
                        raise Exception(
                            f"Video generation {i+1} failed: {prediction.error}"
                        )
                    elif prediction.status == "canceled":
                        raise Exception(f"Video generation {i+1} was canceled")
                    
                    # Wait before next check
                    time.sleep(2)
                
                # Download the result
                if hasattr(prediction.output, 'read'):
                    # Output is a FileOutput object
                    with open(video_path, "wb") as file:
                        file.write(prediction.output.read())
                    
                    generated_videos.append(video_path)
                    generation_info.append({
                        "model": model,
                        "prompt": single_prompt,
                        "image_input": str(single_image),
                        "aspect_ratio": aspect_ratio,
                        "format": output_format,
                        "status": "generated successfully",
                        "prediction_id": prediction.id,
                        "replicate_url": None,
                        "elapsed_time": int(time.time() - start_time)
                    })
                    
                elif isinstance(prediction.output, list) and len(prediction.output) > 0:
                    # Output is a list of URLs
                    video_url = prediction.output[0]
                    
                    # Download the video
                    response = requests.get(video_url)
                    response.raise_for_status()
                    
                    # Save the video
                    with open(video_path, "wb") as file:
                        file.write(response.content)
                    
                    generated_videos.append(video_path)
                    generation_info.append({
                        "model": model,
                        "prompt": single_prompt,
                        "image_input": str(single_image),
                        "aspect_ratio": aspect_ratio,
                        "format": output_format,
                        "status": "generated successfully",
                        "prediction_id": prediction.id,
                        "replicate_url": video_url,
                        "elapsed_time": int(time.time() - start_time)
                    })
                    
                elif isinstance(prediction.output, str):
                    # Output is a direct URL string
                    video_url = prediction.output
                    
                    # Download the video
                    response = requests.get(video_url)
                    response.raise_for_status()
                    
                    # Save the video
                    with open(video_path, "wb") as file:
                        file.write(response.content)
                    
                    generated_videos.append(video_path)
                    generation_info.append({
                        "model": model,
                        "prompt": single_prompt,
                        "image_input": str(single_image),
                        "aspect_ratio": aspect_ratio,
                        "format": output_format,
                        "status": "generated successfully",
                        "prediction_id": prediction.id,
                        "replicate_url": video_url,
                        "elapsed_time": int(time.time() - start_time)
                    })
                    
                else:
                    raise ValueError(
                        f"Unexpected output format from Replicate: "
                        f"{type(prediction.output)}"
                    )
                
            except Exception as e:
                # Add empty path and error info for failed generation
                generated_videos.append("")
                generation_info.append({
                    "model": model,
                    "prompt": single_prompt if 'single_prompt' in locals() else "",
                    "image_input": str(single_image) if 'single_image' in locals() else "",
                    "error": str(e),
                    "status": "generation failed"
                })
        
        return {
            "generated_video_paths": generated_videos,
            "video_generation_info": generation_info
        }

    async def _execute(self, input: Message) -> Message:
        """
        Execute the Replicate video generation using the Polymind framework's Message system.
        
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