"""
Image regeneration pipeline.

Command-line tool for regenerating images:
1. Analyze original image using image understanding
2. Generate new image based on analysis and user preferences

Usage:
    python image_regen_pipeline.py --image-path <image_path> 
    --user-interests <user_interests>

Example:
    python image_regen_pipeline.py --image-path \
    ./examples/media-gen/integration_tests/test_image.png \
    --user-interests "Users like cute cats and capybara"
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict

from pathlib import Path

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, continue without it
    pass

from pipeline import MediaGenerationPipeline, PipelineStep, PipelineStepExecutor
from tools.image_understanding_tool import ImageUnderstandingTool
from tools.replicate_image_gen import ReplicateImageGen


def expand_path(path: str) -> str:
    """Expand path with ~ to user's home directory."""
    return os.path.expanduser(path)


class MediaRegenerationPipeline(MediaGenerationPipeline):
    """
    Pipeline for regenerating media (currently images).
    
    Workflow:
    1. Analyze original image using image understanding
    2. Generate new image based on analysis and user preferences
    """

    system_prompt: str = """
    Please analyze the content of the image, and we want to create a new 
    image generation prompt, combining the original image and the user 
    preferences.
    
    For the generated image generation prompt, follow the below requirement:
    1. The image generation prompt should be as close as possible to the 
       original image, but organically combine the user preferences.
    2. Not just replicate the object and scene in the original image, but 
       also understanding the image style, lighting, and composition.
    3. The generated prompt should not be long (<100 words) and can be 
       easily understood by the image generation model.
    4. The output should be a JSON object with the following fields:
        {
            "image_generation_prompt": "The image generation prompt"
        }
    The user preferences are as below:
    """
    
    def __init__(
        self,
        image_understanding_tool: ImageUnderstandingTool,
        image_generation_tool: Any,  # BaseTool type
        name: str = "media_regeneration",
        debug: bool = False
    ):
        """
        Initialize the media regeneration pipeline.
        
        Args:
            image_understanding_tool: Tool for analyzing images
            image_generation_tool: Tool for generating images
            name: Pipeline name
            debug: Enable debug output
        """
        super().__init__(name)
        self.debug = debug
        
        # Add image understanding step
        self.add_step(
            PipelineStep(
                name="image_understanding",
                tool=image_understanding_tool,
                input_mapping={
                    "original_image": "images",
                    "system_prompt": "prompt",
                    "user_preferences": "user_preferences"
                },
                output_mapping={
                    "analysis": "image_analysis"
                },
                transform_input=self._combine_prompt_and_preferences,
                transform_output=self._extract_analysis
            )
        )
        
        # Add image generation step
        self.add_step(
            PipelineStep(
                name="image_generation",
                tool=image_generation_tool,
                input_mapping={
                    "image_analysis": "prompt",
                    "output_folder": "output_folder",
                    "aspect_ratio": "aspect_ratio",
                    "output_format": "output_format"
                },
                output_mapping={
                    "image_path": "image_path",
                    "generation_info": "generation_info"
                },
                transform_input=self._use_analysis_as_prompt
            )
        )
    
    def regenerate(
        self,
        image_path: str,
        user_interests: str,
        output_folder: str = "~/Downloads",
        aspect_ratio: str = "1:1",
        output_format: str = "png"
    ) -> Dict[str, Any]:
        """
        Regenerate an image based on the original.
        
        Args:
            image_path: Path to the original image
            user_interests: User preferences for regeneration
            output_folder: Folder to save generated image 
                          (default: ~/Downloads)
            aspect_ratio: Aspect ratio for generated image
            output_format: Output format for generated image
            
        Returns:
            Dictionary containing:
                - generated_image_path: Path to the generated image
                - image_analysis: Analysis from image understanding
                - generation_metadata: Additional generation info
        """
        # Prepare input data
        input_data = {
            "original_image": [image_path],
            "system_prompt": self.system_prompt,
            "user_preferences": user_interests,
            "output_folder": output_folder,
            "aspect_ratio": aspect_ratio,
            "output_format": output_format,
            "return_json": True  # Request JSON output from image understanding
        }
        
        # Run pipeline
        result = self.run(input_data)
        
        return result
    
    def _combine_prompt_and_preferences(
        self, tool_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine system prompt with user preferences."""
        system_prompt = tool_input.get("prompt", "")
        user_preferences = tool_input.get("user_preferences", "")
        
        if user_preferences:
            combined_prompt = (
                f"{system_prompt}\n\n\t {user_preferences}"
            )
        else:
            combined_prompt = system_prompt
        
        # Debug: Print the prompt being sent to image understanding
        if self.debug:
            print("\n🔍 DEBUG - Image Understanding Prompt:")
            print(f"   {combined_prompt}")
        
        return {**tool_input, "prompt": combined_prompt}
    
    def _extract_analysis(
        self, tool_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract analysis text and image generation prompt from JSON output.
        
        The image understanding tool returns JSON with the structure:
        {
            "image_generation_prompt": "The image generation prompt"
        }
        
        Handles responses that may be wrapped in markdown code blocks.
        """
        analysis = tool_output.get("analysis", "")
        
        # Debug: Print the raw analysis received from image understanding
        if self.debug:
            print("\n🔍 DEBUG - Raw Image Understanding Analysis:")
            print(f"   {analysis}")
        
        # Clean the analysis text - remove markdown code block markers
        cleaned_analysis = analysis.strip()
        if cleaned_analysis.startswith("```json"):
            cleaned_analysis = cleaned_analysis[7:]  # Remove ```json
        if cleaned_analysis.startswith("```"):
            cleaned_analysis = cleaned_analysis[3:]  # Remove ```
        if cleaned_analysis.endswith("```"):
            cleaned_analysis = cleaned_analysis[:-3]  # Remove trailing ```
        cleaned_analysis = cleaned_analysis.strip()
        
        # Try to parse as JSON and extract the image_generation_prompt
        try:
            analysis_json = json.loads(cleaned_analysis)
            if (isinstance(analysis_json, dict) and
                    "image_generation_prompt" in analysis_json):
                extracted_prompt = analysis_json["image_generation_prompt"]
                
                # Debug: Print the extracted prompt
                if self.debug:
                    print("\n🔍 DEBUG - Extracted Image Generation Prompt:")
                    print(f"   {extracted_prompt}")
                
                return {"analysis": extracted_prompt}
            else:
                # JSON parsed but doesn't have expected structure
                if self.debug:
                    print("\n⚠️  DEBUG - JSON parsed but missing "
                          "image_generation_prompt field")
                return {"analysis": cleaned_analysis}
        except json.JSONDecodeError:
            # Not valid JSON, use the cleaned analysis
            if self.debug:
                print("\n⚠️  DEBUG - Analysis is not valid JSON, "
                      "using cleaned text")
            return {"analysis": cleaned_analysis}
    
    def _use_analysis_as_prompt(
        self, tool_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use the image analysis (which already incorporates user preferences)
        as the final generation prompt.
        
        The image understanding tool generates a prompt that combines
        the original image content with user preferences, so we use
        that directly without further modification.
        """
        # Debug: Print all tool input keys
        if self.debug:
            print("\n🔍 DEBUG - All Tool Input Keys:")
            for key, value in tool_input.items():
                print(f"   {key}: {value}")
        
        # The image_analysis has been mapped to "prompt" by the input mapping
        final_prompt = tool_input.get("prompt", "")
        
        # Debug: Print what we received
        if self.debug:
            print("\n🔍 DEBUG - Image Generation Input:")
            print(f"   prompt (from image_analysis): {final_prompt}")
        
        # Use the prompt directly as the final prompt
        # The image understanding tool already incorporates user preferences
        
        # Debug: Print the prompt being sent to image generation
        if self.debug:
            print("\n🎨 DEBUG - Image Generation Prompt:")
            print(f"   {final_prompt}")
        
        return {**tool_input, "prompt": final_prompt}

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the pipeline with the given input.
        
        Args:
            input_data: Initial input data for the pipeline
            
        Returns:
            Final output from the last step
        """
        # Expand paths
        expanded_image_path = expand_path(input_data["original_image"][0])
        expanded_output_folder = expand_path(input_data["output_folder"])
        
        # Create output directory if it doesn't exist
        Path(expanded_output_folder).mkdir(parents=True, exist_ok=True)
        
        # Update input data with expanded paths
        input_data["original_image"] = [expanded_image_path]
        input_data["output_folder"] = expanded_output_folder
        
        # Debug: Show initial input
        if self.debug:
            print("\n🔍 DEBUG - Initial Pipeline Input:")
            for key, value in input_data.items():
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
                    print(f"   {key}: {value}")
            
            # Execute the step
            executor = PipelineStepExecutor(step)
            step_output = executor.execute(current_input)
            
            # Debug: Show output from this step
            if self.debug:
                print(f"\n🔍 DEBUG - Output from step {i+1} ({step.name}):")
                for key, value in step_output.items():
                    print(f"   {key}: {value}")
            
            # Merge step output with current input for next step
            current_input.update(step_output)
            
            self.logger.debug(f"Step {step.name} output: {step_output}")
        
        self.logger.info("Pipeline execution completed")
        
        # Debug: Show final result
        if self.debug:
            print("\n🔍 DEBUG - Final Pipeline Result Keys:")
            for key, value in current_input.items():
                if key == "image_analysis":
                    print(f"   {key}: {str(value)[:100]}...")
                else:
                    print(f"   {key}: {value}")
        
        return current_input


def setup_logging():
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Regenerate an image based on user interests"
    )
    parser.add_argument(
        "--image-path",
        required=True,
        help="Path to the original image file (supports ~ for home directory)"
    )
    parser.add_argument(
        "--user-interests",
        required=True,
        help="User interests/preferences for regeneration"
    )
    parser.add_argument(
        "--output-folder",
        default="~/Downloads",
        help="Output folder for generated image (default: ~/Downloads)"
    )
    parser.add_argument(
        "--aspect-ratio",
        default="1:1",
        help="Aspect ratio for generated image (default: 1:1)"
    )
    parser.add_argument(
        "--output-format",
        default="png",
        help="Output format for generated image (default: png)"
    )
    parser.add_argument(
        "--generator",
        choices=["replicate"],
        default="replicate",
        help="Image generation service to use (default: replicate)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output to see prompts used in each step"
    )
    
    args = parser.parse_args()
    
    # Expand paths
    expanded_image_path = expand_path(args.image_path)
    
    # Validate input image exists
    if not Path(expanded_image_path).exists():
        print(f"Error: Image file not found: {expanded_image_path}")
        sys.exit(1)
    
    # Setup logging
    setup_logging()
    
    try:
        # Initialize tools
        image_understanding = ImageUnderstandingTool()
        
        # Initialize image generation tool
        image_gen = ReplicateImageGen()
        
        # Create pipeline
        pipeline = MediaRegenerationPipeline(
            image_understanding_tool=image_understanding,
            image_generation_tool=image_gen,
            debug=args.debug
        )
        
        # Regenerate image
        result = pipeline.regenerate(
            image_path=args.image_path,
            user_interests=args.user_interests,
            output_folder=args.output_folder,
            aspect_ratio=args.aspect_ratio,
            output_format=args.output_format
        )
        
        # Output results
        print("\n" + "="*60)
        print("🎨 MEDIA REGENERATION COMPLETE")
        print("="*60)
        
        # Get image path from the result
        generated_path = result.get('image_path', '')
        if generated_path:
            print(f"📁 Image stored at: {generated_path}")
            # Show relative path if it's in Downloads
            downloads_path = os.path.expanduser("~/Downloads")
            if generated_path.startswith(downloads_path):
                relative_path = os.path.relpath(generated_path, downloads_path)
                print(f"   📂 Relative to Downloads: {relative_path}")
        else:
            print("❌ No image path returned")
        
        print("\n🔍 Image Analysis:")
        analysis = result.get('image_analysis', '')
        if analysis:
            print(f"   {analysis}")
        else:
            print("   No analysis available")
        
        print("\n🎯 Final Generation Prompt:")
        # Get the final prompt from generation info
        generation_info = result.get('generation_info', {})
        final_prompt = generation_info.get('prompt', '')
        if final_prompt:
            print(f"   {final_prompt}")
        else:
            print("   No generation prompt available")
        
        if generation_info:
            print("\n📊 Generation Info:")
            # Remove the prompt from metadata to avoid duplication
            filtered_info = {
                k: v for k, v in generation_info.items() if k != 'prompt'
            }
            for key, value in filtered_info.items():
                print(f"   {key}: {value}")
        
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 