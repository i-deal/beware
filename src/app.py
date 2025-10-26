from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import google.generativeai as genai
import os
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from cnn_model import load_model
from model_utils import stream_to_tensor, gemini_query
load_dotenv()

# gemini
genai.configure(api_key=os.environ["GEMINI_KEY"])
model = genai.GenerativeModel("gemini-2.5-pro")

# real-time classifier:
classifier_model = load_model()

app = FastAPI()

@app.post("/generate")
async def generate_response(
    prompt: str = Form(...),
    images: list[UploadFile] = File(None)
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
