"""
Media Generation Demo Package for Polymind framework.

This package demonstrates media generation capabilities including
image and video generation tools that can be integrated into
Polymind agents.
"""

from .tools import (
    ImageGenerationTool,
    VideoGenerationTool
)

__all__ = [
    "ImageGenerationTool",
    "VideoGenerationTool"
]