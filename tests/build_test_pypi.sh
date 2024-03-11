#!/bin/bash

# Exit in case of error
set -e

# Ensure the script is run from the project root directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found. Are you in the right directory?"
    exit 1
fi

# Step 1: Install Poetry
echo "Installing Poetry..."
python -m pip install --upgrade pip
pip install poetry

# Step 2: Check for the POETRY_TEST_PYPI_API_TOKEN environment variable
if [ -z "${POETRY_TEST_PYPI_API_TOKEN}" ]; then
    echo "Error: POETRY_TEST_PYPI_API_TOKEN is not set."
    exit 1
fi

# Step 3: Build the package using Poetry
echo "Building the package with Poetry..."
poetry build

# Step 4: Publish the package to Test PyPI using Poetry
# Make sure to set the environment variable before calling this script or within this script
echo "Publishing the package to Test PyPI with Poetry..."
export POETRY_PYPI_TOKEN_TESTPYPI="${POETRY_TEST_PYPI_API_TOKEN}"
poetry publish --repository testpypi

echo "Package uploaded successfully."
