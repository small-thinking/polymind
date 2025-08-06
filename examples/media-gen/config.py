"""
Configuration management for media generation tools.

This module handles loading environment variables and providing
a centralized configuration interface for the media generation tools.
"""

import os
from typing import Optional
from dotenv import load_dotenv


class MediaGenConfig:
    """Configuration class for media generation tools."""
    
    def __init__(self, env_file: str = ".env"):
        """
        Initialize configuration.
        
        Args:
            env_file: Path to the .env file to load
        """
        # Load environment variables from .env file
        load_dotenv(env_file)
        
        # API Keys
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.openai_org_id: Optional[str] = os.getenv("OPENAI_ORG_ID")
        self.anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
        self.stability_api_key: Optional[str] = os.getenv("STABILITY_API_KEY")
        self.replicate_api_token: Optional[str] = os.getenv(
            "REPLICATE_API_TOKEN"
        )
        
        # Configuration
        self.default_image_model: str = os.getenv(
            "DEFAULT_IMAGE_MODEL", "dall-e-3"
        )
        self.default_video_model: str = os.getenv(
            "DEFAULT_VIDEO_MODEL", "stable-video-diffusion"
        )
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    def validate_api_keys(self) -> dict[str, bool]:
        """
        Validate that required API keys are present.
        
        Returns:
            Dictionary mapping API name to availability status
        """
        return {
            "openai": bool(self.openai_api_key),
            "anthropic": bool(self.anthropic_api_key),
            "stability": bool(self.stability_api_key),
            "replicate": bool(self.replicate_api_token),
        }
    
    def get_missing_keys(self) -> list[str]:
        """
        Get list of missing API keys.
        
        Returns:
            List of missing API key names
        """
        available = self.validate_api_keys()
        return [key for key, available in available.items() if not available]
    
    def print_status(self) -> None:
        """Print configuration status."""
        print("Media Generation Configuration Status:")
        print("=" * 40)
        
        api_status = self.validate_api_keys()
        for api, available in api_status.items():
            status = "✓ Available" if available else "✗ Missing"
            print(f"{api.capitalize()}: {status}")
        
        print(f"\nDefault Image Model: {self.default_image_model}")
        print(f"Default Video Model: {self.default_video_model}")
        print(f"Log Level: {self.log_level}")
        
        missing = self.get_missing_keys()
        if missing:
            print(f"\nMissing API keys: {', '.join(missing)}")
            print("Please check your .env file and ensure all required keys "
                  "are set.")


# Global configuration instance
config = MediaGenConfig() 