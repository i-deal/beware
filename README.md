<div align="center">
    <img src="https://raw.githubusercontent.com/spineda1208/watchout/main/.github/images/logo.png" width="100rem" height="100rem"/>
</div>

A machine learning service for violence and crime detection in images and video frames. This FastAPI-based service uses a custom CNN model to classify images for potential criminal activity and integrates with Google's Gemini AI for detailed analysis.

## Features

- Real-time image classification for violence/crime detection
- CNN-based deep learning model with recurrent architecture
- Integration with Google Gemini 2.5 Pro for contextual analysis
- RESTful API endpoints for image classification and AI-powered analysis
- Support for JPEG/PNG image formats

## Requirements

- Python 3.13 or higher
- uv >= 0.9.5

## Getting Started

### Option 1: Using uv run (Recommended)

This approach doesn't require activating a virtual environment manually. uv handles everything for you.

1. Clone the repository:

```bash
git clone <repository-url>
cd watchout-ml
```

2. Install dependencies:

```bash
uv sync
```

3. Set up environment variables:
   Create a `.env` file in the root directory:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

4. Run the development server:

```bash
uv run fastapi dev app/main.py
```

5. Or run the production server:

```bash
uv run fastapi run app/main.py
```

### Option 2: Using activated virtual environment

If you prefer to activate the virtual environment manually:

1. Clone the repository:

```bash
git clone <repository-url>
cd watchout-ml
```

2. Sync dependencies:

```bash
uv sync
```

3. Set up environment variables:
   Create a `.env` file in the root directory:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

4. Activate the virtual environment:

```bash
source .venv/bin/activate
```

5. Run the development server:

```bash
fastapi dev app/main.py
```

6. Or run the production server:

```bash
fastapi run app/main.py
```

## API Endpoints

### POST /classify

Classifies an image for potential criminal activity.

**Request:**

- Content-Type: `multipart/form-data`
- Body: `image` (file) - JPEG or PNG image

**Response:**

```json
{
  "illicit": true,
  "image_info": {
    "filename": "example.jpg",
    "size_bytes": 123456
  }
}
```

### POST /stream_to

Analyzes images using Google Gemini AI with custom prompts.

**Request:**

- Content-Type: `multipart/form-data`
- Body:
  - `prompt` (string) - Analysis prompt
  - `images` (files, optional) - One or more images

**Response:**

```json
{
  "response": "Analysis result from Gemini AI"
}
```

## Development

### Code Formatting

Format your code using uv:

```bash
uv format
```

### Adding Dependencies

To add a new package:

```bash
uv add <package-name>
```

## Docker Deployment

_(Documentation coming soon)_

## Model Architecture

The service uses a custom ViolenceCNN model with:

- Convolutional feature extraction layers
- Recurrent processing for temporal information
- Binary classification (Violence/Crime vs Normal)
- Trained on the DCSASS Dataset

## License

[Add your license here]
