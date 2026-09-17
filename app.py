"""
app.py

Sprint 3: User Interface and Portability
Ref: cosi_design_doc.md, Section 4 - Subsystem D (WebUI)

A lightweight Flask dashboard that:
    - Serves a search bar (templates/index.html) with real-time autocompletion.
    - Accepts search queries and calls cosi_engine.search() for ranked results.
    - Displays ranked paragraph cards: source file path, blended score,
      and raw text snippet.
    - Exposes a "Refresh Index" action that triggers offline_ingest.ingest()
      (the "Prompted Refresh" mitigation strategy -- see cosi_project_management.pptx,
      Slide 3, "Data Synchronization" risk).

Runs on port 8000 to match the Dockerfile EXPOSE directive.
"""

from flask import Flask, jsonify, render_template, request

import cosi_engine
import offline_ingest

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
    """
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify([])

    results = cosi_engine.search(query, top_n=5)
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
    stats = offline_ingest.ingest()
    return jsonify(stats)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
