FROM python:3.13-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

RUN apt-get update && apt-get install -y gcc libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY pyproject.toml uv.lock* ./

# Install the application dependencies.
RUN uv sync --frozen --no-cache

# Copy application contents after so dependencies cache is not invalidated

# Final image not containing gcc and other build deps
FROM python:3.13-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY --from=builder /src/.venv ./.venv
COPY ./app ./app

ENV PATH="/src/.venv/bin:$PATH"

CMD ["fastapi", "run", "app/main.py", "--port", "80", "--host", "0.0.0.0"]
