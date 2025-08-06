#!/usr/bin/env python3
"""
Setup script for Media Generation Example (Linux).
Handles virtual environment creation and environment configuration.
"""
import subprocess
import sys

import shutil
from pathlib import Path


def run_command(
    cmd: list[str], check: bool = True
) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(
        cmd, check=check, capture_output=True, text=True
    )


def check_python_version() -> None:
    """Check if Python version is compatible."""
    if sys.version_info < (3, 10):
        print("Error: Python 3.10+ is required")
        sys.exit(1)
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"✓ Python {version} detected")


def create_virtual_environment() -> None:
    """Create virtual environment."""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✓ Virtual environment already exists")
        response = input("Do you want to recreate it? (y/N): ").lower()
        if response == 'y':
            shutil.rmtree(venv_path)
        else:
            return
    
    print("Creating virtual environment...")
    run_command([sys.executable, "-m", "venv", "venv"])
    print("✓ Virtual environment created")


def install_requirements() -> None:
    """Install requirements in the virtual environment."""
    pip_path = "venv/bin/pip"
    
    print("Upgrading pip...")
    run_command([pip_path, "install", "--upgrade", "pip"])
    
    print("Installing requirements...")
    run_command([pip_path, "install", "-r", "requirements.txt"])
    print("✓ Requirements installed")


def setup_environment_file() -> None:
    """Set up the .env file from template."""
    env_file = Path(".env")
    example_file = Path("env.example")
    
    if env_file.exists():
        print("✓ .env file already exists")
        response = input("Do you want to overwrite it? (y/N): ").lower()
        if response != 'y':
            return
    
    if example_file.exists():
        shutil.copy(example_file, env_file)
        print("✓ Created .env file from template")
    else:
        print("✗ env.example file not found")
        return


def main() -> None:
    """Main setup function."""
    print("Media Generation Example Setup")
    print("=" * 40)
    
    # Check Python version
    check_python_version()
    
    # Create virtual environment
    create_virtual_environment()
    
    # Install requirements
    install_requirements()
    
    # Setup environment file
    setup_environment_file()
    
    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Edit the .env file with your actual API keys")
    print("2. Activate the virtual environment:")
    print("   source venv/bin/activate")
    print("3. Run 'python example_usage.py' to test the setup")
    print("\nRequired API keys:")
    print("- OPENAI_API_KEY: For DALL-E image generation")
    print("- REPLICATE_API_TOKEN: For various AI models")


if __name__ == "__main__":
    main() 