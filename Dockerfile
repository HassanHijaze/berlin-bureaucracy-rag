FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as a non-root user. The code stays owned by root (read-only for the
# app); only the vector cache is writable, so embeddings can be rebuilt.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data/cache \
    && chown -R app:app /app/data/cache

USER app

EXPOSE 8000

CMD ["uvicorn", "berlin_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
