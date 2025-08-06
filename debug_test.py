#!/usr/bin/env python3
"""Debug script to test the is_valid_image_url function."""

import os
import sys
sys.path.insert(0, '..')

from polymind.core.utils import is_valid_image_url

path = '/Users/Yexi/source/polymind/examples/media-gen/integration_tests/test_image.png'

print(f"Path: {path}")
print(f"os.sep: {repr(os.sep)}")
print(f"Contains sep: {os.sep in path}")
print(f"Starts with http: {path.startswith(('http://', 'https://'))}")
print(f"Condition: {('\\' in path or os.sep in path) and not path.startswith(('http://', 'https://'))}")
print(f"Is URL: {is_valid_image_url(path)}") 