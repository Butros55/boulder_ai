FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=${PORT:-9090} \
    WORKERS=1 \
    THREADS=8

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /api
COPY requirements-linux.txt .

RUN pip install --upgrade pip && pip install -r requirements-linux.txt

COPY . .

EXPOSE 9090
EXPOSE 5000