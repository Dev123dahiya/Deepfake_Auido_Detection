FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8000
ENV MODEL_PATH=outputs/deepfake_detector_enhanced_final.h5

EXPOSE 8000

CMD ["sh", "-c", "uvicorn serve_api:app --host 0.0.0.0 --port ${PORT}"]

