"""
config.py

Central runtime configuration for cosi, read from environment variables so the
same code runs unchanged on the host and inside the Docker container.

Why this exists: inside a container, `localhost` refers to the container itself,
not the machine Ollama is running on. Hard-coding http://localhost:11434 makes
every embedding call fail in Docker, and cosi then silently degrades to
keyword-only search. Making the URL configurable fixes that.

Variables (all optional):
    COSI_OLLAMA_URL      Base URL of the Ollama server.
                         Default: http://localhost:11434 (correct on the host).
                         Docker (Mac/Windows): http://host.docker.internal:11434
    COSI_EMBED_MODEL     Embedding model name. Default: nomic-embed-text
    COSI_OLLAMA_TIMEOUT  Seconds to wait for an embedding response. Default: 30
    COSI_BIND            Address Flask listens on. Default: 0.0.0.0
    COSI_PORT            Port Flask listens on. Default: 8000
    COSI_DEBUG           "1" enables Flask debug mode (never in containers). Default: 0

Note: deliberately NOT named OLLAMA_HOST -- that variable is read by the Ollama
server itself (often set to 0.0.0.0), so reusing it here would cause collisions.
"""

import os


def _normalize_url(raw):
    url = raw.strip().rstrip("/")
    if "://" not in url:
        url = "http://" + url
    return url


OLLAMA_URL = _normalize_url(os.environ.get("COSI_OLLAMA_URL", "http://localhost:11434"))
EMBED_MODEL = os.environ.get("COSI_EMBED_MODEL", "nomic-embed-text")

EMBED_URL = f"{OLLAMA_URL}/api/embeddings"
TAGS_URL = f"{OLLAMA_URL}/api/tags"

OLLAMA_TIMEOUT = float(os.environ.get("COSI_OLLAMA_TIMEOUT", "30"))
OLLAMA_PING_TIMEOUT = 2.0      # startup / health ping should fail fast
OLLAMA_STATUS_TTL = 30.0       # seconds to trust a cached up/down result

BIND_HOST = os.environ.get("COSI_BIND", "0.0.0.0")
PORT = int(os.environ.get("COSI_PORT", "8000"))
DEBUG = os.environ.get("COSI_DEBUG", "0") == "1"
