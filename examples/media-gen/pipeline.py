"""
Modular pipeline system for media generation.

This module provides a flexible pipeline architecture for chaining media
generation tools together. It supports static pipelines with configurable
input/output mappings and allows easy addition of new pipeline steps.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from polymind.core.tool import BaseTool


@dataclass
class PipelineStep:
    """
    Represents a single step in a media generation pipeline.
    
    Attributes:
        name: Unique identifier for the step
        tool: The tool to execute in this step
        input_mapping: Mapping from pipeline input to tool input
        output_mapping: Mapping from tool output to pipeline output
        transform_input: Optional function to transform input before passing to tool
        transform_output: Optional function to transform output after tool execution
    """
    name: str
    tool: BaseTool
    input_mapping: Dict[str, str]
    output_mapping: Dict[str, str]
    transform_input: Optional[
        Callable[[Dict[str, Any]], Dict[str, Any]]
    ] = None
    transform_output: Optional[
        Callable[[Dict[str, Any]], Dict[str, Any]]
    ] = None


class PipelineStepExecutor:
    """
    Executes a single pipeline step with input/output transformations.
    """
    
    def __init__(self, step: PipelineStep):
        """Initialize the step executor."""
        self.step = step
        self.logger = logging.getLogger(f"PipelineStepExecutor.{step.name}")
    
    def execute(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the pipeline step.
        
        Args:
            pipeline_input: Current pipeline input state
            
        Returns:
            Tool output mapped to pipeline output format
        """
        self.logger.info(f"Executing step: {self.step.name}")
        
        # Map pipeline input to tool input
        tool_input = self._map_input(pipeline_input)
        
        # Apply input transformation if provided
        if self.step.transform_input:
            tool_input = self.step.transform_input(tool_input)
        
        self.logger.debug(f"Tool input: {tool_input}")
        
        # Execute the tool
        tool_output = self.step.tool.run(tool_input)
        
        self.logger.debug(f"Tool output: {tool_output}")
        
        # Apply output transformation if provided
        if self.step.transform_output:
            tool_output = self.step.transform_output(tool_output)
        
        # Map tool output to pipeline output
        pipeline_output = self._map_output(tool_output)
        
        self.logger.info(f"Step {self.step.name} completed successfully")
        return pipeline_output
    
    def _map_input(self, pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
        """Map pipeline input to tool input using input_mapping."""
        tool_input = {}
        for pipeline_key, tool_key in self.step.input_mapping.items():
            if pipeline_key in pipeline_input:
                tool_input[tool_key] = pipeline_input[pipeline_key]
        return tool_input
    
    def _map_output(self, tool_output: Dict[str, Any]) -> Dict[str, Any]:
        """Map tool output to pipeline output using output_mapping."""
        pipeline_output = {}
        for tool_key, pipeline_key in self.step.output_mapping.items():
            if tool_key in tool_output:
                pipeline_output[pipeline_key] = tool_output[tool_key]
        return pipeline_output


class MediaGenerationPipeline:
    """
    A modular pipeline for media generation tasks.
    
    This pipeline allows chaining multiple tools together with configurable
    input/output mappings and transformations. It supports static execution
    where each step's output becomes the next step's input.
    """
    
    def __init__(self, name: str):
        """
        Initialize the pipeline.
        
        Args:
            name: Name of the pipeline for logging and identification
        """
        self.name = name
        self.steps: List[PipelineStep] = []
        self.logger = logging.getLogger(f"MediaGenerationPipeline.{name}")
    
    def add_step(self, step: PipelineStep) -> 'MediaGenerationPipeline':
        """
        Add a step to the pipeline.
        
        Args:
            step: PipelineStep to add
            
        Returns:
            Self for method chaining
        """
        self.steps.append(step)
        self.logger.info(f"Added step: {step.name}")
        return self
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the pipeline with the given input.
        
        Args:
            input_data: Initial input data for the pipeline
            
        Returns:
            Final output from the last step
        """
        self.logger.info(f"Starting pipeline execution with {len(self.steps)} steps")
        
        current_input = input_data.copy()
        
        for i, step in enumerate(self.steps):
            self.logger.info(f"Executing step {i+1}/{len(self.steps)}: {step.name}")
            
            # Execute the step
            executor = PipelineStepExecutor(step)
            step_output = executor.execute(current_input)
            
            # Merge step output with current input for next step
            current_input.update(step_output)
            
            self.logger.debug(f"Step {step.name} output: {step_output}")
        
        self.logger.info("Pipeline execution completed")
        return current_input
    
    def get_step_names(self) -> List[str]:
        """Get the names of all steps in the pipeline."""
        return [step.name for step in self.steps] 