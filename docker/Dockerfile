FROM python:3.11-slim-bullseye

# =========================================================
# Java 11 - required by PySpark
# =========================================================

RUN apt-get update && \
    apt-get install -y \
        openjdk-11-jre-headless \
        curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# =========================================================
# Environment
# =========================================================

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# =========================================================
# Python dependencies
# =========================================================

COPY requirements.txt .

RUN pip install \
    --no-cache-dir \
    -r requirements.txt

# =========================================================
# API
# =========================================================

COPY src/serve/api.py ./src/serve/api.py

# =========================================================
# Port
# =========================================================

EXPOSE 8000

# =========================================================
# Start FastAPI
# =========================================================

CMD ["uvicorn", "src.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]