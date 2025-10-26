#!/usr/bin/env python3
"""
Simple test script for the /classify endpoint.
Usage: uv run tests/test_classify.py
"""

import requests
import os
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()

# ============================================
# PLUG IN YOUR IMAGE PATH HERE:
IMAGE_PATH = SCRIPT_DIR / "test.jpeg"
# ============================================


def test_classify_endpoint(
    image_path: Path = IMAGE_PATH, url: str = "http://127.0.0.1:8000/classify"
):
    """Test the /classify endpoint with an image file."""
    print(f"Testing classify endpoint with image: {image_path}")

    try:
        with open(image_path, "rb") as f:
            files = {"image": (str(image_path), f, "image/jpeg")}
            response = requests.post(url, files=files)

        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")

    except FileNotFoundError:
        print(f"Error: Image file not found: {image_path}")
        print(f"Please ensure test.jpeg exists in the tests/ directory")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_classify_endpoint()
