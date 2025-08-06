#!/bin/bash

# Setup script for Media Generation Example
# This script creates a virtual environment and installs dependencies

set -e  # Exit on any error

echo "Setting up Media Generation Example environment..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "To activate the environment in the future:"
echo "  source venv/bin/activate"
echo ""
echo "To run the example:"
echo "  python example_usage.py"
echo ""
echo "To run tests:"
echo "  cd tests && python test_dummy_media_gen.py" 