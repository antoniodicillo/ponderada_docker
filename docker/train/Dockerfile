FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir kagglehub pandas==3.0.6 numpy scikit-learn==1.9.1 joblib==1.6.0
COPY train.py .
ENV MODEL_DIR=/artefatos
CMD ["python", "train.py"]
