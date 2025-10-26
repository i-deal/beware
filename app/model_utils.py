import torch
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms
import numpy as np
import re
import json

# Define transforms to match the model's expected input
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

def post_to_email_server(notif_json: dict):
    import requests
    notif_json['emails'] =["jaeminbird@gmail.com"]
    base_url = "https://trywatchout.tech/api/gemini"
    
    try:
        response = requests.post(
            base_url,
            json=notif_json,
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error sending notification: {str(e)}")
        return None


def clean_and_parse_analysis(analysis_string: str) -> dict:       
    match = re.search(r"```json\s*(.*?)\s*```", analysis_string, re.S)
    
    if match:
        # Extract the content inside the code fences
        raw_json_string = match.group(1).strip()
        
        try:
            # Parse the cleaned string into a Python dictionary
            parsed_analysis = json.loads(raw_json_string)
            
        except json.JSONDecodeError as e:
            # Log error if the inner string is not valid JSON
            print(f"Error parsing embedded JSON: {e}")
            print(f"Attempted to parse: {raw_json_string}")
            # Keep the raw string in the dictionary if parsing fails
            
    return parsed_analysis

def gemini_query(client, images: list) -> str:
    from google.genai import types
    
    prompt = "This is a set of frames from CCTV footage that was flagged for potential criminal activity, quickly analyze if there are any crimes being committed in the footage, then in JSON format with crimeTypeID as one of the following outputs (ignore the <>): <No criminal activity detected>, <Robbery detected>, <Violence detected>, <Shoplifting detected>, <Drug abuse detected>, <Arson detected>. The write a 2 sentence summary of the events in the Summary field."
    
    # Convert PIL images to Part objects
    contents = [prompt]
    for img in images:
        contents.append(img)
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents
    )
    return clean_and_parse_analysis(response.text)
