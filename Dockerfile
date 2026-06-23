FROM python:3.11.15 AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY algear/ algear/
COPY models/ models/
COPY pyproject.toml .
COPY README.md .

# Install the algear package
RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "algear.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
