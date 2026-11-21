# Small, reproducible image for the REST API.
# Offline by default; set HPRAG_LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY to use Claude.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HPRAG_LOG_FORMAT=json

WORKDIR /app

# Install dependencies first so code changes don't invalidate this layer.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[api,anthropic]"

COPY data ./data

# Build the index at image build time so the container starts instantly.
RUN hprag ingest data/corpus --index /app/index --log-level WARNING

# Run as a non-root user.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["hprag", "serve", "--index", "/app/index", "--host", "0.0.0.0", "--port", "8000"]
