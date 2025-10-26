from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import google.generativeai as genai
import os
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
import time
from app.logger import logger

logger.info("Starting FastAPI application...")

load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-pro")

logger.info("Loading ML classifier model...")
start_time = time.time()

from app.cnn_model import load_model
from app.model_utils import stream_to_tensor, gemini_query

classifier_model = load_model()
load_time = time.time() - start_time
logger.info(f"ML classifier model loaded in {load_time:.2f} seconds")

app = FastAPI()
logger.info("FastAPI app ready")


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
