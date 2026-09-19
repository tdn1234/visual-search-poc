# Build context is the project root (see docker-compose.yml), so this
# Dockerfile can COPY both backend/ and catalog/ while preserving the
# same relative layout the app expects on a bare-metal checkout.
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so this layer is cached unless
# requirements.txt actually changes.
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    # PyPI's default 'torch' wheel bundles ~2GB of NVIDIA CUDA
    # libraries we don't need (this POC is CPU-only). Installing from
    # PyTorch's dedicated CPU index first keeps the image small and
    # the build fast on a normal laptop.
    && pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ backend/
COPY catalog/ catalog/

WORKDIR /app/backend

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
