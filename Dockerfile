FROM python:3.12-slim

WORKDIR /app

# pdf extra pulls in MarkItDown for full-text grounding; api for the server
COPY pyproject.toml README.md LICENSE ./
COPY citens ./citens
RUN pip install --no-cache-dir ".[api,pdf]"

# runs/, papers/, data/ live outside the image
VOLUME ["/app/runs", "/app/papers", "/app/data"]
ENV PAPERS_DIR=/app/papers OUTPUT_DIR=/app/runs SJR_CSV_PATH=/app/data/sjr/sjr.csv

EXPOSE 8000
CMD ["uvicorn", "citens.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
