FROM python:3.13-slim-bookworm AS builder

WORKDIR /src

COPY pyproject.toml ./

# --- BUILDER STAGE: Add necessary C++ build dependencies for image libraries ---
RUN apt-get update && \
    # 1. Install gcc and crucial development headers for image processing (libjpeg, libpng)
    apt-get install -y --no-install-recommends gcc libjpeg-dev libpng-dev && \
    # 2. Install torch and torchvision together for version compatibility
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    # 3. Install remaining dependencies
    pip install --no-cache-dir fastapi[standard] numpy pillow google-genai python-dotenv tqdm requests && \
    # 4. Clean up the build tools
    apt-get purge -y gcc libjpeg-dev libpng-dev && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------
# Final stage
FROM python:3.13-slim-bookworm

# --- FINAL STAGE: Install runtime libraries ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # Fix: libjpeg-turbo8 is not available; using the standard runtime library: libjpeg62-turbo
    libjpeg62-turbo \
    libpng16-16 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Crucial Fix #1: Copy the Python site-packages (libraries)
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
# Crucial Fix #2: Copy the executables (uvicorn, fastapi)
COPY --from=builder /usr/local/bin /usr/local/bin

# FIX: Copy model checkpoint files. This assumes the local path is ./app/checkpoints/
# (relative to the Dockerfile) and copies them to the expected container path: /app/checkpoints
COPY ./app/checkpoints /app/checkpoints

COPY ./app ./app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
