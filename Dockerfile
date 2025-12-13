FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and notebooks
COPY src/ src/
COPY notebook/ notebook/
COPY run.sh run.sh

# Create a directory for data (to be mounted)
RUN mkdir -p /app/data
RUN chmod +x /app/run.sh || true

CMD ["bash", "/app/run.sh"]