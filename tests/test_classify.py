#!/usr/bin/env python3
"""
Simple test script for the /classify endpoint.
Usage: python test_classify.py
"""
import requests

# ============================================
# PLUG IN YOUR IMAGE PATH HERE:
IMAGE_PATH = "path/to/your/image.jpg"
# ============================================

def test_classify_endpoint(image_path: str = IMAGE_PATH, url: str = "http://127.0.0.1:8000/classify"):
    """Test the /classify endpoint with an image file."""
    print(f"Testing classify endpoint with image: {image_path}")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': (image_path, f, 'image/jpeg')}
            response = requests.post(url, files=files)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
    except FileNotFoundError:
        print(f"Error: Image file not found: {image_path}")
        print(f"Please update IMAGE_PATH in {__file__}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_classify_endpoint()
