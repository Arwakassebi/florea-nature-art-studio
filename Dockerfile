# FLOREA Lite (CPU). Construire :  docker build -t florea-lite .   Lancer :  docker run -p 7860:7860 florea-lite
FROM python:3.12-slim
WORKDIR /app
COPY requirements-lite.txt .
RUN pip install --no-cache-dir -r requirements-lite.txt
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .
ENV FLOREA_HOST=0.0.0.0 PORT=7860 GRADIO_ANALYTICS_ENABLED=False
EXPOSE 7860
CMD ["python", "-m", "florea.lite"]
