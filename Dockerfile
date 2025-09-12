# Python base
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# System deps:
# - build-essential, python3-dev: build wheels (PyAudio often needs compile)
# - portaudio19-dev (brings libportaudio2): required for PyAudio
# - libasound2, libasound2-dev: ALSA (audio) runtime/headers for PyAudio
# - ffmpeg: optional but useful for audio handling in SR pipelines
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    portaudio19-dev \
    libasound2 \
    libasound2-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Copy deps first for caching
COPY requirements.txt /app/requirements.txt

# Faster, more reliable builds for packages with native parts
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . /app

# Streamlit default port
EXPOSE 8501

# Run the Streamlit UI (change app.py if your entrypoint differs)
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
