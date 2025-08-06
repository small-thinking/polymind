"""
Tests for the OpenAI image generation tool.
"""

import os
import sys
from unittest.mock import Mock, patch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tools.openai_image_gen import OpenAIImageGen


class TestOpenAIImageGen:
    """Test cases for OpenAIImageGen tool."""
    
    def test_init(self):
        """Test tool initialization."""
        tool = OpenAIImageGen()
        assert tool.tool_name == "openai_image_generator"
        assert len(tool.descriptions) == 3
    
    def test_input_spec(self):
        """Test input parameter specification."""
        tool = OpenAIImageGen()
        params = tool.input_spec()
        
        # Check that all expected parameters are present
        param_names = [p.name for p in params]
        expected_params = ["prompt", "aspect_ratio", "output_format", "image_path"]
        
        for param in expected_params:
            assert param in param_names
    
    def test_output_spec(self):
        """Test output parameter specification."""
        tool = OpenAIImageGen()
        params = tool.output_spec()
        
        # Check that all expected parameters are present
        param_names = [p.name for p in params]
        expected_params = ["image_path", "generation_info"]
        
        for param in expected_params:
            assert param in param_names
    
    @patch('tools.openai_image_gen.OpenAI')
    def test_run_success(self, mock_openai):
        """Test successful image generation."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_output = Mock()
        mock_output.type = "image_generation_call"
        mock_output.result = "base64_encoded_image_data"
        mock_response.output = [mock_output]
        
        mock_client = Mock()
        mock_client.responses.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Create tool and test
        tool = OpenAIImageGen()
        result = tool.run({
            "prompt": "Test image",
            "image_path": "/tmp/test.png"
        })
        
        # Verify result
        assert result["image_path"] == "/tmp/test.png"
        assert result["generation_info"]["model"] == "gpt-4o-mini"
        assert result["generation_info"]["status"] == "generated successfully"
    
    @patch('tools.openai_image_gen.OpenAI')
    def test_run_failure(self, mock_openai):
        """Test image generation failure."""
        # Mock OpenAI to raise exception
        mock_client = Mock()
        mock_client.responses.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client
        
        # Create tool and test
        tool = OpenAIImageGen()
        result = tool.run({
            "prompt": "Test image"
        })
        
        # Verify error handling
        assert result["image_path"] == ""
        assert result["generation_info"]["status"] == "generation failed"
        assert "API Error" in result["generation_info"]["error"]
    
    def test_default_parameters(self):
        """Test default parameter values."""
        tool = OpenAIImageGen()
        
        # Test with minimal input
        with patch.object(tool, 'client') as mock_client:
            mock_response = Mock()
            mock_output = Mock()
            mock_output.type = "image_generation_call"
            mock_output.result = "base64_encoded_image_data"
            mock_response.output = [mock_output]
            mock_client.responses.create.return_value = mock_response
            
            tool.run({"prompt": "Test"})
            
            # Verify default values were used
            call_args = mock_client.responses.create.call_args
            tool_params = call_args[1]['tools'][0]['parameters']
            
            assert tool_params['size'] == "1024x1024"
            assert tool_params['quality'] == "low"
            assert tool_params['format'] == "png"
            assert tool_params['compression'] == 80
            assert tool_params['background'] == "opaque"
    
    @patch('tools.openai_image_gen.OpenAI')
    def test_custom_parameters(self, mock_openai):
        """Test custom parameter values."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_output = Mock()
        mock_output.type = "image_generation_call"
        mock_output.result = "base64_encoded_image_data"
        mock_response.output = [mock_output]
        
        mock_client = Mock()
        mock_client.responses.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Create tool and test with custom parameters
        tool = OpenAIImageGen()
        tool.run({
            "prompt": "Test image",
            "size": "1024x1536",
            "quality": "high",
            "output_format": "jpg",
            "compression": 95,
            "background": "transparent"
        })
        
        # Verify custom values were used
        call_args = mock_client.responses.create.call_args
        tool_params = call_args[1]['tools'][0]['parameters']
        
        assert tool_params['size'] == "1024x1536"
        assert tool_params['quality'] == "high"
        assert tool_params['format'] == "jpg"
        assert tool_params['compression'] == 95
        assert tool_params['background'] == "transparent" 