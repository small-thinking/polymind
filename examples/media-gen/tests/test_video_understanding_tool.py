"""
Tests for the VideoUnderstandingTool.

This module contains unit tests for the VideoUnderstandingTool class,
testing its ability to extract screenshots from videos and generate
coherent image generation prompts.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch

from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.video_understanding_tool import VideoUnderstandingTool
from utils.video_utils import ScreenshotInfo


class TestVideoUnderstandingTool(unittest.TestCase):
    """Test cases for VideoUnderstandingTool."""

    def setUp(self):
        """Set up test fixtures."""
        self.api_key = "test_api_key"
        self.tool = VideoUnderstandingTool(api_key=self.api_key)

    def test_init(self):
        """Test tool initialization."""
        self.assertEqual(self.tool.api_key, self.api_key)
        self.assertEqual(self.tool.model, "gpt-4o-mini")
        self.assertIsNotNone(self.tool.client)

    def test_init_without_api_key(self):
        """Test initialization without API key raises error."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                VideoUnderstandingTool()

    def test_input_spec(self):
        """Test input specification."""
        params = self.tool.input_spec()
        
        # Check required parameters
        param_names = [p.name for p in params]
        self.assertIn("video_path", param_names)
        self.assertIn("user_preference", param_names)
        self.assertIn("screenshot_interval", param_names)
        self.assertIn("output_dir", param_names)
        self.assertIn("max_tokens", param_names)
        
        # Check video_path is required
        video_path_param = next(p for p in params if p.name == "video_path")
        self.assertTrue(video_path_param.required)

    def test_output_spec(self):
        """Test output specification."""
        params = self.tool.output_spec()
        
        # Check output parameters
        param_names = [p.name for p in params]
        self.assertIn("image_prompts", param_names)
        self.assertIn("scene_descriptions", param_names)
        self.assertIn("screenshot_paths", param_names)
        self.assertIn("metadata", param_names)

    @patch('examples.media-gen.tools.video_understanding_tool.extract_screenshots')
    def test_extract_screenshots(self, mock_extract):
        """Test screenshot extraction."""
        # Mock screenshot data
        mock_screenshots = [
            ScreenshotInfo(
                frame_number=0,
                timestamp=0.0,
                timestamp_str="00:00:00",
                file_path="/tmp/screenshot_1.jpg"
            ),
            ScreenshotInfo(
                frame_number=60,
                timestamp=2.0,
                timestamp_str="00:00:02",
                file_path="/tmp/screenshot_2.jpg"
            )
        ]
        mock_extract.return_value = mock_screenshots
        
        # Test extraction
        result = self.tool._extract_screenshots(
            "/path/to/video.mp4", 2.0, "/tmp/output"
        )
        
        self.assertEqual(len(result), 2)
        mock_extract.assert_called_once_with(
            video_path="/path/to/video.mp4",
            interval_seconds=2.0,
            output_dir="/tmp/output",
            filename_prefix="video_scene"
        )

    @patch('examples.media-gen.tools.video_understanding_tool.encode_image_to_base64')
    @patch('examples.media-gen.tools.video_understanding_tool.OpenAI')
    def test_analyze_screenshots(self, mock_openai, mock_encode):
        """Test screenshot analysis."""
        # Mock screenshots
        screenshots = [
            ScreenshotInfo(
                frame_number=0,
                timestamp=0.0,
                timestamp_str="00:00:00",
                file_path="/tmp/screenshot_1.jpg"
            )
        ]
        
        # Mock base64 encoding
        mock_encode.return_value = "base64_encoded_image"
        
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''
        {
            "scenes": [
                {
                    "scene_number": 1,
                    "timestamp": "00:00:00",
                    "prompt": "A cat sitting on a windowsill",
                    "description": "Scene 1: Cat on windowsill",
                    "key_elements": ["cat", "windowsill"]
                }
            ]
        }
        '''
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        # Test analysis
        result = self.tool._analyze_screenshots(screenshots, "cinematic style")
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["prompt"], "A cat sitting on a windowsill")

    @patch('examples.media-gen.tools.video_understanding_tool.extract_screenshots')
    @patch('examples.media-gen.tools.video_understanding_tool.encode_image_to_base64')
    @patch('examples.media-gen.tools.video_understanding_tool.OpenAI')
    def test_run_success(self, mock_openai, mock_encode, mock_extract):
        """Test successful tool execution."""
        # Mock file existence
        with patch('os.path.exists', return_value=True):
            # Mock screenshots
            screenshots = [
                ScreenshotInfo(
                    frame_number=0,
                    timestamp=0.0,
                    timestamp_str="00:00:00",
                    file_path="/tmp/screenshot_1.jpg"
                ),
                ScreenshotInfo(
                    frame_number=60,
                    timestamp=2.0,
                    timestamp_str="00:00:02",
                    file_path="/tmp/screenshot_2.jpg"
                )
            ]
            mock_extract.return_value = screenshots
            
            # Mock base64 encoding
            mock_encode.return_value = "base64_encoded_image"
            
            # Mock OpenAI response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = '''
            {
                "scenes": [
                    {
                        "scene_number": 1,
                        "timestamp": "00:00:00",
                        "prompt": "A cat sitting on a windowsill",
                        "description": "Scene 1: Cat on windowsill",
                        "key_elements": ["cat", "windowsill"]
                    },
                    {
                        "scene_number": 2,
                        "timestamp": "00:00:02",
                        "prompt": "The cat jumps down and walks",
                        "description": "Scene 2: Cat walking",
                        "key_elements": ["cat", "movement"]
                    }
                ]
            }
            '''
            mock_openai.return_value.chat.completions.create.return_value = mock_response
            
            # Test run
            result = self.tool.run({
                "video_path": "/path/to/video.mp4",
                "user_preference": "cinematic style"
            })
            
            # Verify results
            self.assertIn("image_prompts", result)
            self.assertIn("scene_descriptions", result)
            self.assertIn("screenshot_paths", result)
            self.assertIn("metadata", result)
            
            self.assertEqual(len(result["image_prompts"]), 2)
            self.assertEqual(len(result["scene_descriptions"]), 2)
            self.assertEqual(len(result["screenshot_paths"]), 2)

    def test_run_missing_video_path(self):
        """Test run with missing video path."""
        with self.assertRaises(ValueError):
            self.tool.run({})

    def test_run_video_not_found(self):
        """Test run with non-existent video file."""
        with patch('os.path.exists', return_value=False):
            with self.assertRaises(FileNotFoundError):
                self.tool.run({"video_path": "/nonexistent/video.mp4"})

    @patch('examples.media-gen.tools.video_understanding_tool.extract_screenshots')
    def test_run_no_screenshots(self, mock_extract):
        """Test run when no screenshots are extracted."""
        mock_extract.return_value = []
        
        with patch('os.path.exists', return_value=True):
            with self.assertRaises(RuntimeError):
                self.tool.run({"video_path": "/path/to/video.mp4"})

    def test_tool_name(self):
        """Test tool name property."""
        self.assertEqual(self.tool.tool_name, "video_understanding")

    def test_tool_description(self):
        """Test tool description property."""
        description = self.tool.tool_description
        self.assertIsInstance(description, str)
        self.assertIn("video", description.lower())


if __name__ == '__main__':
    unittest.main()
