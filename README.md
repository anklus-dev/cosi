# cosi

(Pronounced "Cozy"; From **cosi**ne similarity)

A lightweight, blazingly fast, and completely offline search utility for personal notes and markdown documents. This project merges classical discrete mathematics with modern vector embeddings to provide robust hybrid retrieval.

Developed as an extra credit project for **CSC 2510: Theoretical Foundations for Computer Science** under Dr. Suxing Liu.

---

## System Architecture

The project splits execution into two highly optimized pipelines:

```mermaid
flowchart LR
    subgraph offline["Offline Ingestion Pipeline"]
        direction TB
        raw["Raw Notes (.md)"] --> chunker["Chunker"]
        chunker --> stemmer1["Stemmer (Sets)"]
        chunker --> embed["Ollama (Embed)"]
        stemmer1 --> db[("SQLite DB (.db)")]
        embed --> db
    end
```
```mermaid
flowchart LR
    subgraph search["Query"]
        direction TB
        query["Query"] --> stemmer2["Stemmer"]
        query --> ollama2["Ollama"]
        stemmer2 --> bm25["BM25 Matcher"]
        ollama2 --> cosine["Cosine Similarity"]
        bm25 --> ranker["Hybrid Ranker"]
        cosine --> ranker
        ranker --> results["Top Results (WebUI)"]
    end

    db --> ranker
```


---

## Tech Stack & Prerequisites

This is a record of *my* environment while developing this project.

- **Development Environment:** NixOS / Linux
- **Deployment Platform**: Docker (Cross-platform compatability)
- **Runtime:** Python 3.x
- **Libraries:** 
  - `nltk` (Tokenization & stemming baseline)
  - `sqlite3` (Built-in SQL relational database)
  - `numpy` or `scikit-learn` (Optimized matrix calculations for Cosine Similarity)
- **Local AI:** Ollama running `nomic-embed-text` (a fully auditable, open-data text embedding model)

---

## Quick Start & Setup

cosi runs in Docker. Ollama runs separately on your own machine (not in the container), and cosi connects to it.

### 1. Install and start Ollama (on the host)

Install Ollama from <https://ollama.com>, then pull the embedding model:

```bash
ollama pull nomic-embed-text
```

Make sure the Ollama server is running (the desktop app does this automatically; otherwise run `ollama serve`).

> Ollama is optional. Without it, cosi still works in **keyword-only (BM25) mode** and prints a warning at startup.

### 2. Start cosi

**macOS / Windows (Docker Desktop):**

```bash
docker compose up --build
```

**Linux:**

```bash
docker compose -f docker-compose.yml -f docker-compose.linux.yml up --build
```

The Linux variant uses host networking, because Ollama on Linux only listens on `127.0.0.1` by default and a normally-networked container can't reach it.

### 3. Use it

1. Open <http://localhost:8000>.
2. Click **Refresh Index** to index the notes in `notes/` (needed on first run and after editing notes).
3. Search.

Your notes folder (`./notes`) and the index (`./data/cosi.db`) are mounted into the container, so edits to your notes are visible after a refresh and the index survives restarts.

### Checking the connection

- <http://localhost:8000/api/status> shows whether search is running in `hybrid` or `keyword` mode, and why.
- The container logs (`docker compose logs cosi`) show the Ollama check made at startup.

### Without Docker (development)

```bash
pip install -r requirements.txt
python app.py              # COSI_DEBUG=1 python app.py for Flask debug mode
python tests/test_ollama.py
```

### Configuration

All settings are environment variables (see `config.py`):

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `COSI_OLLAMA_URL` | `http://localhost:11434` (container: `http://host.docker.internal:11434`) | Where Ollama is reachable |
| `COSI_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `COSI_OLLAMA_TIMEOUT` | `30` | Seconds to wait for an embedding |
| `COSI_NOTES_DIR` | `./notes` | Folder of `.md` notes to index |
| `COSI_BIND` / `COSI_PORT` | `0.0.0.0` / `8000` | Web server address |
| `COSI_DEBUG` | `0` | `1` enables Flask debug mode (local development only) |

---

## Datasets

- `notes/`: six short travel notes (the default).
- `datasets/meditations/`: Marcus Aurelius' *Meditations* (Meric Casaubon's translation, public domain, Project Gutenberg eBook #2680), one file per book, each passage labeled with its traditional number (e.g. `VIII.48`). Rebuild it from the source text with:

  ```bash
  python tools/prepare_meditations.py datasets/sources/pg2680.txt --out datasets/meditations
  ```

To search the Meditations instead of the travel notes:

```bash
COSI_NOTES_DIR=datasets/meditations python app.py       # then click Refresh Index
```

With Docker, mount it as the notes folder instead: change `./notes:/app/notes` to `./datasets/meditations:/app/notes` in `docker-compose.yml`.

## Evaluation

`eval/run_eval.py` measures how often each search method puts the right passage at the top, using hand-labeled questions (`eval/queries_travel.json`, `eval/queries_meditations.json`). It builds its own temporary index, so `data/cosi.db` is never touched. With Ollama running:

```bash
python eval/run_eval.py                                            # travel notes
python eval/run_eval.py --notes datasets/meditations \
       --queries eval/queries_meditations.json --chunkers hard,sentence
```

It compares keyword-only (BM25), meaning-only (cosine) and cosi's blend, sweeps the blend weight from 0 to 1, compares plain and `search_document:`/`search_query:`-prefixed embeddings, and runs a significance test. Results go to `eval/results/<dataset>/` (`summary.md`, `summary.json`, `per_query.csv`). `--no-embed` runs the keyword branch only, without Ollama.

## Tests

```bash
python -m pytest tests/
```

## Poster

`poster/build_poster.py` fills the GSU poster template (`poster/template.pptx`) from the evaluation results and writes `poster/cosi_poster.pptx`. Run the evaluation first; anything still missing appears as an orange `[pending: ...]` marker. Fonts: Oswald and Fira Sans (free Google fonts).

---

## Developer Log

- **2026-08-29:** Initialized private Git repository. Verified professor's BM25 benchmark script running successfully on NixOS.

