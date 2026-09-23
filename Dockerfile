FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY rag ./rag
COPY app.py ./
RUN pip install --no-cache-dir ".[cloud]"
RUN useradd --create-home rag && mkdir /app/data && chown -R rag:rag /app
USER rag
EXPOSE 8000
CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
