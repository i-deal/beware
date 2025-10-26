import torch

def stream_to_tensor(img_stream) -> torch.Tensor:
    # TODO: implement actual conversion from image to tensor once image format is known
    tensor = torch.tensor(img_stream, dtype=torch.uint8)
    return tensor

def gemini_query(model, images: list) -> str:
    prompt = "This is a set of frames from CCTV footage that was flagged for potential criminal activity, quickly analyze if there are any crimes being committed in the footage, then respond with one of the following outputs (ignore the <>): <No criminal activity detected>, <Robbery detected>, <Violence detected>, <Shoplifting detected>, <Drug abuse detected>, <Arson detected>"
    inputs = [prompt] + images
    response = model.generate_content(inputs)
    return response.text