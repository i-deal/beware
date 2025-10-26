from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import google.generativeai as genai
import os
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
import time
import torch
from app.logger import log

log.info("Starting FastAPI application...")

load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-pro")

log.info("Loading ML classifier model...")
start_time = time.time()

from app.cnn_model import load_model
from app.model_utils import stream_to_tensor, gemini_query

classifier_model = load_model()
load_time = time.time() - start_time
log.info(f"ML classifier model loaded in {load_time:.2f} seconds")

app = FastAPI()
log.info("FastAPI app ready")


@app.post("/stream_to")
async def generate_response(
    prompt: str = Form(...), images: list[UploadFile] = File(None)
):
    inputs = [prompt]

    # Convert uploaded images into Pillow Image objects
    if images:
        for img in images:
            image_data = await img.read()
            pil_img = Image.open(BytesIO(image_data))
            inputs.append(pil_img)

    try:
        response = model.generate_content(inputs)
        return JSONResponse({"response": response.text})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/classify")
async def classify_image(image: UploadFile = File(...)):
    """
    Mock endpoint that accepts a JPEG/PNG image and runs inference with the CNN model.

    Args:
        image: Image file (JPEG/PNG)

    Returns:
        JSON response with classification result and confidence
    """
    try:
        log.info(f"Received image for classification: {image.filename}")
        start_time = time.time()

        # Read image bytes
        image_data = await image.read()
        log.info(f"Image size: {len(image_data)} bytes")

        # Convert to tensor batch
        tensor_batch = stream_to_tensor(image_data, num_frames=15)
        log.info(f"Tensor batch shape: {tensor_batch.shape}")

        # Move to same device as model
        device = next(classifier_model.parameters()).device
        tensor_batch = tensor_batch.to(device)

        # Run inference
        with torch.no_grad():
            output = classifier_model(tensor_batch)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        inference_time = time.time() - start_time
        log.info(f"Inference completed in {inference_time:.3f} seconds")

        # Class 1 = Violence/Crime, Class 0 = Normal
        illicit = predicted_class == 1

        return JSONResponse(
            {
                "illicit": illicit,
                "image_info": {
                    "filename": image.filename,
                    "size_bytes": len(image_data),
                },
            }
        )

    except Exception as e:
        log.error(f"Error during classification: {str(e)}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)
