#!/bin/bash

# Exit in case of error
set -e

# Ensure the script is run from the project root directory
if [ ! -f "setup.py" ] && [ ! -f "pyproject.toml" ]; then
    echo "Error: setup.py or pyproject.toml not found. Are you in the right directory?"
    exit 1
fi

# Step 1: Build the package
echo "Building the package..."
python -m pip install --upgrade pip setuptools wheel twine
python setup.py sdist bdist_wheel

# Step 2: Upload the package to Test PyPI
# Make sure the password is in the .pypirc file
echo "Uploading the package to Test PyPI..."
twine upload --repository testpypi dist/*

echo "Package uploaded successfully."