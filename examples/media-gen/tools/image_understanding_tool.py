"""
Image understanding tool using OpenAI's GPT-4o-mini API.

This module provides a tool for analyzing images using OpenAI's vision 
capabilities. It supports both local image files and image URLs, and can 
return structured JSON responses for easy parsing.
"""

import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from polymind.core.tool import BaseTool, Param
from polymind.core.utils import encode_image_to_base64, is_valid_image_url


class ImageUnderstandingTool(BaseTool):
    """
    Tool for image understanding using OpenAI's GPT-4o-mini API.
    
    This tool can analyze images using natural language prompts and return
    structured responses. It supports both local image files and image URLs.
    """

    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    client: Optional[Any] = None

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini", **kwargs):
        """
        Initialize the image understanding tool.
        
        Args:
            api_key (Optional[str]): OpenAI API key. If None, will use environment variable.
            model (str): OpenAI model to use for image understanding
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
            tool_name="image_understanding",
            descriptions=[
                "Analyze images using OpenAI's GPT-4o-mini vision capabilities",
                "Image understanding and analysis tool",
                "Vision-based AI tool for image description and analysis"
            ],
            api_key=api_key,
            model=model,
            **kwargs
        )
        
        # Initialize the OpenAI client
        self.client = OpenAI(api_key=self.api_key)

    def input_spec(self) -> List[Param]:
        """Define the input parameters for image understanding."""
        return [
            Param(
                name="prompt",
                type="str",
                required=False,
                description="Text prompt describing what to analyze in the image",
                example="What objects do you see in this image? Describe the scene and any text visible."
            ),
            Param(
                name="images",
                type="List[str]",
                required=True,
                description="List of image paths (local files) or URLs to analyze",
                example='["path/to/image.jpg", "https://example.com/image.png"]'
            ),
            Param(
                name="return_json",
                type="bool",
                required=False,
                description="Whether to return response as JSON format",
                example="true"
            ),
            Param(
                name="max_tokens",
                type="int",
                required=False,
                description="Maximum number of tokens in the response",
                example="1000"
            ),

        ]

    def output_spec(self) -> List[Param]:
        """Define the output parameters for image understanding."""
        return [
            Param(
                name="analysis",
                type="str",
                required=True,
                description="Analysis of the image based on the prompt",
                example="The image shows a cat sitting on a windowsill..."
            ),
            Param(
                name="confidence",
                type="float",
                required=False,
                description="Confidence score of the analysis (0.0 to 1.0)",
                example="0.95"
            ),
            Param(
                name="metadata",
                type="Dict[str, str]",
                required=False,
                description="Additional metadata about the analysis",
                example='{"model": "gpt-4o-mini", "tokens_used": "150"}'
            )
        ]

    def _prepare_image_content(self, image_input: str) -> Dict[str, str]:
        """
        Prepare image content for OpenAI API.
        
        Args:
            image_input (str): Image path or URL
            
        Returns:
            Dict[str, str]: Image content dictionary for OpenAI API
        """
        if is_valid_image_url(image_input):
            # It's a URL
            return {
                "type": "image_url",
                "image_url": {"url": image_input}
            }
        else:
            # It's a local file path
            try:
                base64_image = encode_image_to_base64(image_input)
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            except Exception as e:
                raise ValueError(f"Failed to process image {image_input}: {e}")

    def run(self, input: dict) -> dict:
        """
        Analyze images using OpenAI's vision capabilities.
        
        Args:
            input (dict): Dictionary containing:
                - prompt: Text prompt for analysis (optional, default: "What's in this image?")
                - images: List of image paths or URLs
                - return_json: Whether to return JSON response (optional, default: False)
                - max_tokens: Maximum tokens in response (optional, default: 1000)
        
        Returns:
            dict: Dictionary containing:
                - analysis: Text analysis of the image(s)
                - confidence: Confidence score (if available)
                - metadata: Additional metadata
        """
        # Extract parameters
        prompt = input.get("prompt", "What's in this image?")
        images = input.get("images", [])
        return_json = input.get("return_json", False)
        max_tokens = input.get("max_tokens", 1000)
        
        if not images:
            raise ValueError("At least one image must be provided")
        
        # Prepare content for OpenAI API
        content = [{"type": "text", "text": prompt}]
        
        # Add images to content
        for image in images:
            image_content = self._prepare_image_content(image)
            content.append(image_content)
        
        # If JSON response is requested, modify the prompt
        if return_json:
            content[0]["text"] = f"{prompt}\n\nPlease respond with a valid JSON object."
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
                response_format={"type": "json_object"} if return_json else None
            )
            
            analysis = response.choices[0].message.content
            
            # Parse JSON if requested
            if return_json:
                try:
                    analysis_dict = json.loads(analysis)
                    # If it's a structured response, extract main content
                    if isinstance(analysis_dict, dict):
                        if "analysis" in analysis_dict:
                            analysis = analysis_dict["analysis"]
                        elif "description" in analysis_dict:
                            analysis = analysis_dict["description"]
                        elif "content" in analysis_dict:
                            analysis = analysis_dict["content"]
                except json.JSONDecodeError:
                    # If JSON parsing fails, use the raw response
                    pass
            
            # Extract metadata
            metadata = {
                "model": self.model,
                "tokens_used": str(response.usage.total_tokens),
                "prompt_tokens": str(response.usage.prompt_tokens),
                "completion_tokens": str(response.usage.completion_tokens)
            }
            
            return {
                "analysis": analysis,
                "metadata": metadata
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to analyze images: {e}")

    @property
    def tool_name(self) -> str:
        """Return the name of the tool."""
        return "image_understanding"

    @property
    def tool_description(self) -> str:
        """Return the description of the tool."""
        return "Analyze images using OpenAI's GPT-4o-mini vision capabilities with custom prompts" 