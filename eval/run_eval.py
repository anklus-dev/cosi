"""
eval/run_eval.py

Retrieval evaluation for cosi: does hybrid search actually beat keyword-only and
vector-only search, and is the 0.4 / 0.6 blend a good choice?

What it does
    1. Builds a fresh, temporary index of a notes folder (your real data/cosi.db is
       never touched), once per variant. A variant is an embedding style plus a chunker:
           plain     chunks and queries embedded as-is
           prefixed  "search_document: " / "search_query: " prefixes, which
                     nomic-embed-text was trained to expect
           sentence  long paragraphs split on sentence boundaries (cosi's chunker)
           hard      the original chunker: a cut every 500 characters
    2. Runs every query in a labeled query file through:
           bm25      keyword branch only (cosi's Ollama-offline fallback)
           vector    cosine-similarity branch only
           hybrid    alpha * BM25 + (1 - alpha) * cosine, both min-max normalized,
                     for alpha = 0.0, 0.1, ... 1.0  (cosi uses alpha = 0.4)
    3. Scores each ranking against the labels and writes:
           summary.md     tables ready to copy onto a poster
           summary.json   the same numbers, machine-readable
           per_query.csv  rank of the first relevant passage per query and method

Metrics (k = cut-off)
    Hit@1     share of queries whose top result is relevant
    Recall@3  share of the relevant passages that appear in the top 3
    Recall@5  same, top 5
    MRR       mean reciprocal rank: 1/rank of the first relevant result, averaged
              (1.0 = always first, 0.5 = typically second, 0 = never found)
    A passage the keyword branch gives no score to (no shared words) counts as
    not found, which matches what cosi shows the user.

Significance
    A paired randomization test on per-query reciprocal rank compares cosi's hybrid
    against each single branch. p < 0.05 means the difference is unlikely to be luck
    from this particular set of queries.

Labels
    Each query lists its relevant passages either by a text snippet
        {"file": "kyoto.md", "contains": "Fushimi Inari"}
    or by passage number for datasets whose paragraphs start with one (Meditations)
        {"file": "book_08.md", "passage": "VIII.48"}
    A passage split across several chunks counts as found at the rank of its best chunk.

Usage (from the project root, inside `nix develop`, with Ollama running)
    python eval/run_eval.py                                   # travel notes, plain + prefixed
    python eval/run_eval.py --notes datasets/meditations --queries eval/queries_meditations.json \
                            --chunkers hard,sentence          # also compares the two chunkers
    python eval/run_eval.py --variants plain                  # one embedding style only
    python eval/run_eval.py --no-embed                        # BM25 only, no Ollama needed

The weight sweep is tuned on the same queries it is scored on, so the "best alpha"
is a description of this query set, not proof of a universally better weight.
"""

import argparse
import csv
import datetime as dt
import json
import os
import random
import re
import shutil
import sqlite3
import statistics
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cosi_engine  # noqa: E402
import offline_ingest  # noqa: E402
from config import EMBED_MODEL, OLLAMA_URL  # noqa: E402

DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
COSI_ALPHA = cosi_engine.BM25_WEIGHT  # 0.4
ALPHAS = [round(a / 10, 1) for a in range(11)]
METRICS = ["hit@1", "recall@3", "recall@5", "mrr"]

# Words ignored when measuring how many words a query shares with its target passage.
STOPWORDS = set("""a an the and or of to in on at for with by from is are be it its this that
these those i me my we you your how what where when which who why can do does did should would
could there here than then about into over under up down out as so not no just very most more
some any all get best good place places""".split())


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_index(notes_dir, db_path, prefix, embed, chunker):
    """Chunk, stem and (optionally) embed every .md file into a fresh SQLite index."""
    conn = offline_ingest.init_db(db_path)
    markdown, _ = offline_ingest._scan_notes(notes_dir)
    n_chunks = 0
    started = time.perf_counter()
    for rel, path in sorted(markdown.items()):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
        doc_id = conn.execute(
            "INSERT INTO documents (file_path, file_hash, last_modified) VALUES (?, ?, ?)",
            (rel, offline_ingest.hash_file(path), os.path.getmtime(path)),
        ).lastrowid
        for chunk in offline_ingest.chunk_text(raw, mode=chunker):
            vector = None
            if embed:
                vector = offline_ingest.get_embedding(prefix + chunk)
                if vector is None:
                    conn.close()
                    sys.exit(f"Embedding failed while indexing {rel}. Is Ollama running at {OLLAMA_URL}?")
            conn.execute(
                "INSERT INTO chunks (document_id, raw_text, stemmed_tokens, embedding_vector) VALUES (?, ?, ?, ?)",
                (doc_id, chunk, offline_ingest.stem_chunk(chunk), offline_ingest._vector_to_blob(vector)),
            )
            n_chunks += 1
            print(f"\r  indexed {n_chunks} chunks", end="", flush=True)
    conn.commit()
    conn.close()
    print(f"  ({time.perf_counter() - started:.1f}s)")
    return n_chunks


def load_chunks(db_path):
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT chunks.id, documents.file_path, chunks.raw_text FROM chunks "
        "JOIN documents ON documents.id = chunks.document_id"
    ).fetchall()
    conn.close()
    return {cid: {"file": f, "text": t} for cid, f, t in rows}


def resolve_labels(queries, chunks, notes_dir):
    """
    Turn each label into the set of chunk ids that make up that passage, and store
    them as q["relevant_groups"] (one set per label). Fails loudly on bad labels.
    """
    def norm(text):  # notes wrap lines, so compare with all whitespace collapsed
        return re.sub(r"\s+", " ", text).lower().strip()

    passage_cache = {}

    def passage_text(file, label):
        if file not in passage_cache:
            with open(os.path.join(notes_dir, file), encoding="utf-8") as f:
                passage_cache[file] = {
                    p.split(" ", 1)[0]: norm(p)
                    for p in re.split(r"\n\s*\n", f.read()) if p.strip() and not p.startswith("#")
                }
        return passage_cache[file].get(label)

    problems = []
    for q in queries:
        groups = []
        for label in q["relevant"]:
            if "passage" in label:
                text = passage_text(label["file"], label["passage"])
                if text is None:
                    problems.append(f"{q['id']}: no passage {label['passage']} in {label['file']}")
                    continue
                head = text[:80]
                # A chunk belongs to the passage if it starts inside it (the passage was split)
                # or contains the passage's opening (short passages merged into one chunk).
                ids = {cid for cid, c in chunks.items()
                       if c["file"] == label["file"]
                       and (norm(c["text"])[:80] in text or head in norm(c["text"]))}
                what = f"passage {label['passage']}"
            else:
                needle = norm(label["contains"])
                ids = {cid for cid, c in chunks.items()
                       if c["file"] == label["file"] and needle in norm(c["text"])}
                what = repr(label["contains"])
            if not ids:
                problems.append(f"{q['id']}: no chunk in {label['file']} matches {what}")
            groups.append(ids)
        q["relevant_groups"] = groups
    if problems:
        sys.exit("Label errors (fix the query file):\n  " + "\n  ".join(problems))


def word_overlap(query, chunks, groups):
    """Content-word stems the query shares with its relevant passage(s)."""
    def stems(text):
        return {offline_ingest._stemmer.stem(w)
                for w in re.findall(r"\b\w+\b", text.lower()) if w not in STOPWORDS}
    q = stems(query)
    shared = set()
    for cid in set().union(*groups):
        shared |= q & stems(chunks[cid]["text"])
    return sorted(shared)


# ---------------------------------------------------------------------------
# Ranking and scoring
# ---------------------------------------------------------------------------

def rank(scores):
    """Chunk ids ordered by score (desc), ties broken by id for determinism."""
    return [cid for cid, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def hybrid_scores(bm25, cosine, alpha):
    b = cosi_engine.normalize_scores(bm25)
    c = cosi_engine.normalize_scores(cosine)
    return {cid: alpha * b.get(cid, 0.0) + (1 - alpha) * c.get(cid, 0.0) for cid in set(b) | set(c)}


def score_ranking(ranking, groups):
    """A label counts as found at the rank of its best-ranked chunk."""
    positions = {cid: i + 1 for i, cid in enumerate(ranking)}
    ranks = sorted(min(positions[c] for c in g if c in positions)
                   for g in groups if any(c in positions for c in g))
    first = ranks[0] if ranks else None
    return {
        "first_rank": first,
        "hit@1": 1.0 if first == 1 else 0.0,
        "recall@3": sum(r <= 3 for r in ranks) / len(groups),
        "recall@5": sum(r <= 5 for r in ranks) / len(groups),
        "mrr": 1.0 / first if first else 0.0,
    }


def mean_metrics(rows):
    if not rows:
        return {m: None for m in METRICS}
    return {m: statistics.fmean(r[m] for r in rows) for m in METRICS}


def randomization_test(a, b, iterations=10000, seed=2510):
    """Two-sided paired sign-flip test on the mean difference of a and b."""
    diffs = [x - y for x, y in zip(a, b)]
    observed = abs(statistics.fmean(diffs))
    if observed == 0:
        return 1.0
    rng = random.Random(seed)
    extreme = 0
    for _ in range(iterations):
        flipped = statistics.fmean(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(flipped) >= observed - 1e-12:
            extreme += 1
    return (extreme + 1) / (iterations + 1)


def evaluate_variant(name, prefix, chunker, queries, notes_dir, embed, workdir):
    print(f"\n[{name}] building index ...")
    db_path = os.path.join(workdir, f"{name.replace('/', '-')}.db")
    n_chunks = build_index(notes_dir, db_path, prefix, embed, chunker)
    chunks = load_chunks(db_path)
    resolve_labels(queries, chunks, notes_dir)

    per_query = []
    timings = {"bm25_ms": [], "embed_ms": [], "cosine_ms": []}
    for q in queries:
        t0 = time.perf_counter()
        bm25 = cosi_engine.bm25_search(cosi_engine.tokenize_and_stem_query(q["query"]), db_path=db_path)
        t1 = time.perf_counter()
        cosine = {}
        if embed:
            vec = offline_ingest.get_embedding((QUERY_PREFIX if prefix else "") + q["query"])
            if vec is None:
                sys.exit(f"Embedding failed for query {q['id']}. Is Ollama running at {OLLAMA_URL}?")
            t2 = time.perf_counter()
            cosine = cosi_engine.vector_search(vec, db_path=db_path)
            t3 = time.perf_counter()
            timings["embed_ms"].append((t2 - t1) * 1000)
            timings["cosine_ms"].append((t3 - t2) * 1000)
        timings["bm25_ms"].append((t1 - t0) * 1000)

        methods = {"bm25": score_ranking(rank(bm25), q["relevant_groups"])}
        if embed:
            methods["vector"] = score_ranking(rank(cosine), q["relevant_groups"])
            for alpha in ALPHAS:
                methods[f"hybrid@{alpha}"] = score_ranking(rank(hybrid_scores(bm25, cosine, alpha)), q["relevant_groups"])
            methods["cosi"] = methods[f"hybrid@{COSI_ALPHA}"]

        shared = word_overlap(q["query"], chunks, q["relevant_groups"])
        per_query.append({"id": q["id"], "query": q["query"], "category": q["category"],
                          "shared_words": shared, "methods": methods})

    method_names = list(per_query[0]["methods"])
    groups = {
        "all": per_query,
        **{c: [p for p in per_query if p["category"] == c] for c in sorted({p["category"] for p in per_query})},
        "no shared words": [p for p in per_query if not p["shared_words"]],
    }
    summary = {g: {m: mean_metrics([p["methods"][m] for p in rows]) | {"n": len(rows)}
                   for m in method_names}
               for g, rows in groups.items() if rows}

    significance = {}
    if embed:
        rr = {m: [p["methods"][m]["mrr"] for p in per_query] for m in ("cosi", "bm25", "vector")}
        significance = {"cosi vs bm25": randomization_test(rr["cosi"], rr["bm25"]),
                        "cosi vs vector": randomization_test(rr["cosi"], rr["vector"])}

    return {
        "variant": name,
        "chunks": n_chunks,
        "summary": summary,
        "significance": significance,
        "timings_ms": {k: round(statistics.fmean(v), 2) for k, v in timings.items() if v},
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt(x):
    return "–" if x is None else f"{x:.2f}"


def write_reports(results, queries_meta, out_dir, args):
    os.makedirs(out_dir, exist_ok=True)
    lines = [
        f"# cosi retrieval evaluation: {queries_meta.get('name', 'queries')}",
        "",
        f"- Date: {dt.datetime.now():%Y-%m-%d %H:%M}",
        f"- Notes folder: `{args.notes}`  |  Query file: `{args.queries}`",
        f"- Embedding model: `{EMBED_MODEL}`" if not args.no_embed else "- Embeddings: disabled (BM25 only)",
        f"- Queries: {len(results[0]['per_query'])}",
        "",
    ]
    has_vec = "cosi" in results[0]["summary"]["all"]
    lines += ["## Overview (all queries, MRR)", "",
              "| Variant | Chunks | Keyword only | Vector only | cosi hybrid | cosi Hit@1 |" if has_vec else
              "| Variant | Chunks | Keyword only | Keyword Hit@1 |",
              "| :--- | ---: | ---: | ---: | ---: | ---: |" if has_vec else "| :--- | ---: | ---: | ---: |"]
    for res in results:
        a = res["summary"]["all"]
        if has_vec:
            lines.append(f"| {res['variant']} | {res['chunks']} | {fmt(a['bm25']['mrr'])} | {fmt(a['vector']['mrr'])} "
                         f"| {fmt(a['cosi']['mrr'])} | {fmt(a['cosi']['hit@1'])} |")
        else:
            lines.append(f"| {res['variant']} | {res['chunks']} | {fmt(a['bm25']['mrr'])} | {fmt(a['bm25']['hit@1'])} |")
    lines.append("")
    for res in results:
        s = res["summary"]
        lines += [f"## Variant: {res['variant']} ({res['chunks']} chunks)", ""]
        main = ["bm25", "vector", "cosi"] if "cosi" in s["all"] else ["bm25"]
        label = {"bm25": "Keyword only (BM25)", "vector": "Vector only (cosine)",
                 "cosi": f"cosi hybrid ({COSI_ALPHA} / {round(1 - COSI_ALPHA, 1)})"}
        for group, rows in s.items():
            lines += [f"### {group} (n = {rows['bm25']['n']})", "",
                      "| Method | Hit@1 | Recall@3 | Recall@5 | MRR |", "| :--- | ---: | ---: | ---: | ---: |"]
            for m in main:
                r = rows[m]
                lines.append(f"| {label[m]} | {fmt(r['hit@1'])} | {fmt(r['recall@3'])} | {fmt(r['recall@5'])} | {fmt(r['mrr'])} |")
            lines.append("")

        if "cosi" in s["all"]:
            lines += ["### Weight sweep (all queries)", "",
                      "alpha = weight on BM25; 1 - alpha = weight on cosine. alpha 0 ≈ vector only, alpha 1 ≈ keyword only.", "",
                      "| alpha | Hit@1 | Recall@3 | MRR |", "| ---: | ---: | ---: | ---: |"]
            best = max(ALPHAS, key=lambda a: (s["all"][f"hybrid@{a}"]["mrr"], -abs(a - COSI_ALPHA)))
            for a in ALPHAS:
                r = s["all"][f"hybrid@{a}"]
                tag = (" ← cosi" if a == COSI_ALPHA else "") + (" ← best MRR" if a == best else "")
                lines.append(f"| {a:.1f}{tag} | {fmt(r['hit@1'])} | {fmt(r['recall@3'])} | {fmt(r['mrr'])} |")
            lines += ["", "### Significance (paired randomization test on reciprocal rank)", ""]
            for k, p in res["significance"].items():
                lines.append(f"- {k}: p = {p:.4f}" + ("  (significant at 0.05)" if p < 0.05 else ""))

            wins = [p for p in res["per_query"]
                    if p["methods"]["cosi"]["first_rank"] == 1 and p["methods"]["bm25"]["first_rank"] != 1]
            if wins:
                lines += ["", "### Queries cosi gets right that keyword search misses (good demo material)", "",
                          "| Query | BM25 rank | cosi rank | Shared words |", "| :--- | ---: | ---: | :--- |"]
                for p in wins:
                    br = p["methods"]["bm25"]["first_rank"]
                    lines.append(f"| {p['query']} | {br if br else 'not found'} | 1 | {', '.join(p['shared_words']) or 'none'} |")
            losses = [p for p in res["per_query"] if p["methods"]["cosi"]["first_rank"] != 1]
            if losses:
                lines += ["", "### Queries where cosi's top result is wrong (look at these)", "",
                          "| Query | cosi rank | BM25 rank | Vector rank |", "| :--- | ---: | ---: | ---: |"]
                for p in losses:
                    m = p["methods"]
                    lines.append(f"| {p['query']} | {m['cosi']['first_rank'] or '–'} | "
                                 f"{m['bm25']['first_rank'] or 'not found'} | {m['vector']['first_rank'] or '–'} |")

        t = res["timings_ms"]
        lines += ["", "### Average time per query", ""] + [f"- {k.replace('_ms', '')}: {v} ms" for k, v in t.items()] + [""]

    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines))

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"meta": {"queries": args.queries, "notes": args.notes, "model": EMBED_MODEL},
                   "results": [{k: v for k, v in r.items() if k != "per_query"} for r in results]}, f, indent=2)

    with open(os.path.join(out_dir, "per_query.csv"), "w", newline="") as f:
        w = csv.writer(f)
        methods = [m for m in results[0]["per_query"][0]["methods"] if not m.startswith("hybrid@")]
        w.writerow(["variant", "id", "category", "query", "shared_words"] + [f"{m}_rank" for m in methods])
        for r in results:
            for p in r["per_query"]:
                w.writerow([r["variant"], p["id"], p["category"], p["query"], " ".join(p["shared_words"])]
                           + [p["methods"][m]["first_rank"] or "" for m in methods])

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Evaluate cosi retrieval quality.")
    ap.add_argument("--notes", default=os.path.join(ROOT, "notes"), help="folder of .md notes to index")
    ap.add_argument("--queries", default=os.path.join(ROOT, "eval", "queries_travel.json"))
    ap.add_argument("--out", default=None, help="output folder (default: eval/results/<query set>)")
    ap.add_argument("--variants", default="plain,prefixed",
                    help="comma list of embedding styles: plain, prefixed (default: both)")
    ap.add_argument("--chunkers", default="sentence",
                    help="comma list of chunkers: sentence, hard (default: sentence)")
    ap.add_argument("--no-embed", action="store_true", help="skip Ollama; evaluate BM25 only")
    args = ap.parse_args()

    with open(args.queries) as f:
        meta = json.load(f)
    queries = meta["queries"]

    embed = not args.no_embed
    if embed:
        ok, reason = offline_ingest.ollama_available()
        if not ok:
            sys.exit(f"{reason}\nStart Ollama, or pass --no-embed for a BM25-only run.")

    styles = [v.strip() for v in args.variants.split(",") if v.strip()] if embed else ["plain"]
    chunkers = [c.strip() for c in args.chunkers.split(",") if c.strip()]
    bad = [v for v in styles if v not in ("plain", "prefixed")] + [c for c in chunkers if c not in ("sentence", "hard")]
    if bad:
        sys.exit(f"Unknown variant/chunker: {', '.join(bad)}")
    variants = [(f"{style}/{chunker}", DOC_PREFIX if style == "prefixed" else "", chunker)
                for chunker in chunkers for style in styles]

    out_dir = args.out or os.path.join(ROOT, "eval", "results", meta.get("name", "queries"))
    workdir = tempfile.mkdtemp(prefix="cosi-eval-")
    try:
        results = [evaluate_variant(name, prefix, chunker, queries, args.notes, embed, workdir)
                   for name, prefix, chunker in variants]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    report = write_reports(results, meta, out_dir, args)
    print("\n" + report)
    print(f"\nSaved summary.md, summary.json and per_query.csv to {out_dir}")


if __name__ == "__main__":
    main()
