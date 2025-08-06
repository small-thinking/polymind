"""
Tests for the Replicate image generation tool.
"""

import os
import sys
from unittest.mock import Mock, patch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from tools.replicate_image_gen import ReplicateImageGen
except ImportError as e:
    if "replicate" in str(e):
        print("❌ Replicate package not installed. Please install it with:")
        print("   pip install replicate")
        sys.exit(1)
    else:
        raise


class TestReplicateImageGen:
    """Test cases for ReplicateImageGen tool."""
    
    def test_init(self):
        """Test tool initialization."""
        tool = ReplicateImageGen()
        assert tool.tool_name == "replicate_image_generator"
        assert len(tool.descriptions) == 3
        assert tool.model == "prunaai/wan-2.2-image"
    
    def test_init_custom_model(self):
        """Test tool initialization with custom model."""
        tool = ReplicateImageGen("stability-ai/sdxl")
        assert tool.model == "stability-ai/sdxl"
    
    def test_input_spec(self):
        """Test input parameter specification."""
        tool = ReplicateImageGen()
        params = tool.input_spec()
        
        # Check that all expected parameters are present
        param_names = [p.name for p in params]
        expected_params = ["prompt", "aspect_ratio", "output_format", "output_folder"]
        
        for param in expected_params:
            assert param in param_names
    
    def test_output_spec(self):
        """Test output parameter specification."""
        tool = ReplicateImageGen()
        params = tool.output_spec()
        
        # Check that all expected parameters are present
        param_names = [p.name for p in params]
        expected_params = ["image_path", "generation_info"]
        
        for param in expected_params:
            assert param in param_names
    
    @patch('tools.replicate_image_gen.replicate')
    def test_run_success(self, mock_replicate):
        """Test successful image generation."""
        # Mock Replicate response
        mock_output = Mock()
        mock_output.read.return_value = b"fake_image_data"
        mock_output.url.return_value = "https://replicate.delivery/.../output.jpeg"
        
        mock_replicate.run.return_value = mock_output
        
        # Create tool and test
        tool = ReplicateImageGen()
        result = tool.run({
            "prompt": "Test image",
            "output_folder": "/tmp"
        })
        
        # Verify result
        assert result["image_path"] != ""
        assert result["generation_info"]["model"] == "prunaai/wan-2.2-image"
        assert result["generation_info"]["status"] == "generated successfully"
    
    @patch('tools.replicate_image_gen.replicate')
    def test_run_failure(self, mock_replicate):
        """Test image generation failure."""
        # Mock Replicate to raise exception
        mock_replicate.run.side_effect = Exception("API Error")
        
        # Create tool and test
        tool = ReplicateImageGen()
        result = tool.run({
            "prompt": "Test image"
        })
        
        # Verify error handling
        assert result["image_path"] == ""
        assert result["generation_info"]["status"] == "generation failed"
        assert "API Error" in result["generation_info"]["error"]
    
    def test_default_parameters(self):
        """Test default parameter values."""
        tool = ReplicateImageGen()
        
        # Test with minimal input
        with patch.object(tool, 'run') as mock_run:
            mock_run.return_value = {
                "image_path": "/tmp/test.jpg",
                "generation_info": {"status": "success"}
            }
            
            tool.run({"prompt": "Test"})
            
            # Verify default values were used
            call_args = mock_run.call_args[0][0]
            assert call_args["aspect_ratio"] == "4:3"
            assert call_args["output_format"] == "jpeg"
            assert "~/Downloads" in call_args["output_folder"]
    
    @patch('tools.replicate_image_gen.replicate')
    def test_custom_parameters(self, mock_replicate):
        """Test custom parameter values."""
        # Mock Replicate response
        mock_output = Mock()
        mock_output.read.return_value = b"fake_image_data"
        mock_output.url.return_value = "https://replicate.delivery/.../output.jpeg"
        
        mock_replicate.run.return_value = mock_output
        
        # Create tool and test with custom parameters
        tool = ReplicateImageGen()
        result = tool.run({
            "prompt": "Test image",
            "seed": 12345,
            "aspect_ratio": "16:9",
            "model": "stability-ai/sdxl"
        })
        
        # Verify custom values were used
        call_args = mock_replicate.run.call_args
        model = call_args[0][0]
        input_params = call_args[1]["input"]
        
        assert model == "stability-ai/sdxl"
        assert input_params["prompt"] == "Test image"
        assert input_params["aspect_ratio"] == "16:9"
        assert input_params["seed"] == 12345 