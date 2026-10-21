"""
Diagnostic utility to verify local Ollama API connectivity
and test nomic-embed-text embedding vector generation.

Honors COSI_OLLAMA_URL (default http://localhost:11434), so the same check
works on the host and inside the container:

    python tests/test_ollama.py
    docker compose exec -T cosi python - < tests/test_ollama.py
"""

import json
import os
import urllib.request

OLLAMA_URL = os.environ.get("COSI_OLLAMA_URL", "http://localhost:11434").rstrip("/")
EMBED_MODEL = os.environ.get("COSI_EMBED_MODEL", "nomic-embed-text")


def get_embedding(text):
    """
    Retrieve vector coordinates from the local Ollama embeddings endpoint.

    Args:
        text (str): The text segment to be embedded.

    Returns:
        list[float] or None: 768-dimensional float vector, or None if connection fails.
    """
    url = f"{OLLAMA_URL}/api/embeddings"
    data = {
        "model": EMBED_MODEL,
        "prompt": text
    }

    # Use built-in urllib to avoid external package dependencies
    req_body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=req_body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            res_body = json.loads(response.read().decode("utf-8"))
            return res_body.get("embedding")
    except Exception as e:
        print(f"Error connecting to Ollama at {OLLAMA_URL}: {e}")
        return None


if __name__ == "__main__":
    test_sentence = "Discrete mathematics is the foundation of computer science."
    vector = get_embedding(test_sentence)

    if vector:
        print("Success!")
        print(f"Ollama URL: {OLLAMA_URL}")
        print(f"Vector Dimensions: {len(vector)}")
        print(f"First 5 coordinates: {vector[:5]}...")
    else:
        print(f"Failed to fetch embedding from {OLLAMA_URL}. "
              f"Ensure Ollama is serving ('ollama serve') and the model is pulled "
              f"('ollama pull {EMBED_MODEL}').")
