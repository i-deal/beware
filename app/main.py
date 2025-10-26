import warnings

# Suppress Pydantic warnings from google-genai package
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
import os
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
import time
import torch
from app.logger import log

log.info("Starting FastAPI application...")

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

log.info("Loading ML classifier model...")
start_time = time.time()

from app.cnn_model import load_model
from app.model_utils import gemini_query, transform, post_to_email_server

classifier_model = load_model()
load_time = time.time() - start_time
log.info(f"ML classifier model loaded in {load_time:.2f} seconds")

app = FastAPI()
log.info("FastAPI app ready")

@app.get("/health", response_class=JSONResponse, tags=["System"])
def health_check():
    return {"status": "ok", "message": "Service is healthy and operational"}

@app.post("/stream_to")
async def generate_response(images: list[UploadFile] = File(...)):
    """
    Endpoint that accepts a sequence of video frames, classifies them as a batch for illicit activity,
    and if detected, analyzes them with Gemini for detailed description.
    """
    try:
        log.info(f"Received {len(images)} frames for processing")
        device = next(classifier_model.parameters()).device
        
        pil_images = []
        frame_tensors = []
        
        # process all images into tensors
        for img in images:
            image_data = await img.read()
            pil_img = Image.open(BytesIO(image_data)).convert("RGB") 
            
            pil_images.append(pil_img)
            
            img_tensor = transform(pil_img)
            frame_tensors.append(img_tensor)
            
        tensor_batch = torch.stack(frame_tensors)
        tensor_batch = tensor_batch.to(device)
            
        with torch.no_grad():
            output = classifier_model(tensor_batch)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
        if predicted_class == 0:
            gemini_result = gemini_query(client, pil_images)
            post_to_email_server(gemini_result)
            return JSONResponse({
                "illicit_detected": True,
                "confidence": confidence,
                "analysis": gemini_result
            })
        else:
            return JSONResponse({
                "illicit_detected": False,
                "confidence": confidence,
                "analysis": "No criminal activity detected"
            })
            
    except Exception as e:
        log.error(f"Error during processing: {str(e)}")
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
