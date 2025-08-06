#!/usr/bin/env python3
"""
Setup script for media generation tools environment.

This script helps users set up their .env file with the required API keys.
"""

import shutil

from pathlib import Path


def setup_environment():
    """Set up the environment configuration."""
    print("Media Generation Tools Environment Setup")
    print("=" * 40)
    
    # Check if .env already exists
    env_file = Path(".env")
    example_file = Path("env.example")
    
    if env_file.exists():
        print("✓ .env file already exists")
        response = input("Do you want to overwrite it? (y/N): ").lower()
        if response != 'y':
            print("Setup cancelled.")
            return
    
    # Copy example file to .env
    if example_file.exists():
        shutil.copy(example_file, env_file)
        print("✓ Created .env file from template")
    else:
        print("✗ env.example file not found")
        return
    
    print("\nNext steps:")
    print("1. Edit the .env file with your actual API keys")
    print("2. Run 'python example_usage.py' to test the setup")
    print("\nRequired API keys:")
    print("- OPENAI_API_KEY: For DALL-E image generation")
    print("- REPLICATE_API_TOKEN: For various AI models")
    
    print("\n✓ Environment setup completed!")


if __name__ == "__main__":
    setup_environment() 