# cosi -- local hybrid search engine
# Sprint 3: Portability (cosi_design_doc.md, Section 5)
#
# Ollama is NOT inside this image. It runs on the host machine, and the
# container reaches it through COSI_OLLAMA_URL (see config.py / README).

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    COSI_OLLAMA_URL=http://host.docker.internal:11434 \
    COSI_BIND=0.0.0.0 \
    COSI_PORT=8000

WORKDIR /app

# Dependencies first, so code edits don't invalidate the pip layer.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application code
COPY config.py offline_ingest.py cosi_engine.py app.py ./
COPY templates/ templates/
COPY static/ static/

# Sample notes ship inside the image so a fresh container has something to
# search. Mounting your own folder over /app/notes replaces them.
COPY notes/ notes/

# SQLite index lives here; mount a volume to keep it between runs.
RUN mkdir -p /app/data
VOLUME ["/app/data", "/app/notes"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/status', timeout=4)" || exit 1

CMD ["python", "app.py"]
