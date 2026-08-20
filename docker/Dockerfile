# Use the 'bullseye' Debian release which officially supports Java 11
FROM python:3.11-slim-bullseye

# Install Java 11 (Required for PySpark to run without memory access errors)
RUN apt-get update && \
    apt-get install -y openjdk-11-jre-headless && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API script
COPY src/serve/api.py ./src/serve/api.py

EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "src.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]