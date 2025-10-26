import torch
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms
import numpy as np

# Define transforms to match the model's expected input
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def stream_to_tensor(img_bytes: bytes, num_frames: int = 15) -> torch.Tensor:
    """
    Convert image bytes to a tensor batch suitable for the ViolenceCNN model.

    Args:
        img_bytes: Raw image bytes (JPEG/PNG)
        num_frames: Number of frames to create (model expects 15 frames)

    Returns:
        Tensor of shape [num_frames, 3, 224, 224]
    """
    # Open image from bytes
    img = Image.open(BytesIO(img_bytes)).convert("RGB")

    # Apply transforms
    img_tensor = transform(img)

    # Repeat the same image for num_frames (simulating video frames)
    # Shape: [num_frames, 3, 224, 224]
    batch_tensor = img_tensor.unsqueeze(0).repeat(num_frames, 1, 1, 1)

    return batch_tensor


def gemini_query(client, images: list) -> str:
    from google.genai import types
    
    prompt = "This is a set of frames from CCTV footage that was flagged for potential criminal activity, quickly analyze if there are any crimes being committed in the footage, then respond with one of the following outputs (ignore the <>): <No criminal activity detected>, <Robbery detected>, <Violence detected>, <Shoplifting detected>, <Drug abuse detected>, <Arson detected>"
    
    # Convert PIL images to Part objects
    contents = [prompt]
    for img in images:
        contents.append(types.Part.from_image(img))
    
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=contents
    )
    return response.text
