"""
Diagnostic utility to verify local Ollama API connectivity 
and test nomic-embed-text embedding vector generation.
"""

import json
import urllib.request

def get_embedding(text):
    """
    Retrieve vector coordinates from the local Ollama embeddings endpoint.

    Args:
        text (str): The text segment to be embedded.

    Returns:
        list[float] or None: 768-dimensional float vector, or None if connection fails.
    """
    url = "http://localhost:11434/api/embeddings"
    data = {
        "model": "nomic-embed-text",
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
        with urllib.request.urlopen(req) as response:
            res_body = json.loads(response.read().decode("utf-8"))
            return res_body.get("embedding")
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return None

if __name__ == "__main__":
    test_sentence = "Discrete mathematics is the foundation of computer science."
    vector = get_embedding(test_sentence)
    
    if vector:
        print("Success!")
        print(f"Vector Dimensions: {len(vector)}")
        print(f"First 5 coordinates: {vector[:5]}...")
    else:
        print("Failed to fetch embedding. Ensure Ollama is serving locally ('ollama serve').")
