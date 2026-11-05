"""
app.py

Sprint 3: User Interface and Portability
Ref: cosi_design_doc.md, Section 4 - Subsystem D (WebUI)

A lightweight Flask dashboard that:
    - Serves a search bar (templates/index.html) with real-time autocompletion.
    - Accepts search queries and calls cosi_engine.search() for ranked results.
    - Displays ranked paragraph cards: source file path, blended score,
      BM25 vs. cosine component scores, and raw text snippet.
    - Exposes a "Refresh Index" action that triggers offline_ingest.ingest()
      (the "Prompted Refresh" mitigation strategy -- see cosi_project_management.pptx,
      Slide 3, "Data Synchronization" risk).
    - Exposes /api/status so the UI (and Docker's HEALTHCHECK) can tell whether
      search is running in hybrid or keyword-only fallback mode.

Listens on COSI_BIND:COSI_PORT (default 0.0.0.0:8000, matching the Dockerfile's
EXPOSE directive). Debug mode is off unless COSI_DEBUG=1 -- the Werkzeug debugger
allows arbitrary code execution and must never be reachable from a container port.
"""

import os

from flask import Flask, jsonify, render_template, request

import cosi_engine
import offline_ingest
from config import BIND_HOST, DEBUG, OLLAMA_URL, PORT

app = Flask(__name__)


@app.route("/")
def index():
    """Render the main search page."""
    return render_template("index.html")


@app.route("/api/search")
def api_search():
    """
    Query param: ?q=<search terms>
    Returns JSON list of ranked chunk results from cosi_engine.search().
    The engine's relative-threshold cutoff decides how many results come back
    (up to cosi_engine.MAX_RESULTS_CAP), so no fixed top_n is passed here.
    """
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify([])

    results = cosi_engine.search(query)
    return jsonify(results)


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    """
    Manually trigger re-indexing of the notes directory.
    Corresponds to the "Prompted Refresh" mitigation strategy (user-initiated,
    not an automatic background process). Returns stats (files/chunks indexed,
    and any unsupported file types found in notes/) so the UI can report
    specifics back to the user instead of a bare "done".
    """
    # Re-check Ollama first so a refresh after starting Ollama picks it up immediately.
    status = cosi_engine.check_ollama(force=True)
    stats = offline_ingest.ingest()
    if isinstance(stats, dict):
        stats.setdefault("embeddings_enabled", status["ok"])
        if not status["ok"]:
            stats.setdefault("warning", f"Indexed without embeddings: {status['reason']}")
    return jsonify(stats)


@app.route("/api/status")
def api_status():
    """Report search mode and index size. Also used by the Docker HEALTHCHECK."""
    status = cosi_engine.check_ollama()
    return jsonify({
        "mode": "hybrid" if status["ok"] else "keyword",
        "ollama": status,
        "ollama_url": OLLAMA_URL,
        "index": cosi_engine.corpus_stats(),
    })


def startup_check():
    """Ping handshake from cosi_design_doc.md Section 6 ("Graceful Fallback")."""
    status = cosi_engine.check_ollama(force=True)
    if status["ok"]:
        print(f"[cosi] {status['reason']} -- hybrid search enabled.")
    else:
        print(f"[cosi] WARNING: {status['reason']}")
        print("[cosi] Falling back to keyword-only (BM25) search until Ollama is available.")
        print("[cosi] To enable meaning search: install Ollama from https://ollama.com/download,")
        print("[cosi] run 'ollama pull nomic-embed-text', then click 'Refresh Index'.")

    if not os.path.exists(offline_ingest.DB_PATH):
        print("[cosi] No index found yet -- click 'Refresh Index' in the WebUI to build it.")


if __name__ == "__main__":
    # Under the debug reloader this module runs twice; only check in the serving process.
    if not DEBUG or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        startup_check()
    app.run(host=BIND_HOST, port=PORT, debug=DEBUG)
