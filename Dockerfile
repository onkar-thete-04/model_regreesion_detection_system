FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY llm_regress ./llm_regress
COPY config.yaml .
COPY datasets ./datasets

ENTRYPOINT ["python", "-m", "llm_regress.cli"]
