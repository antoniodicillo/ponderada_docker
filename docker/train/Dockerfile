FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir kagglehub pandas numpy scikit-learn
COPY train.py .
ENV MODEL_DIR=/artefatos
CMD ["python", "train.py"]
