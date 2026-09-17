// script.js
// Handles the search bar's real-time search (debounced fetch to /api/search),
// highlights matching query terms in results, renders the score as a small
// meter, and wires up #refresh-btn to POST /api/refresh and report what changed.

const searchBox = document.getElementById("search-box");
const resultsEl = document.getElementById("results");
const refreshBtn = document.getElementById("refresh-btn");
const refreshStatusEl = document.getElementById("refresh-status");

let debounceTimer = null;

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function escapeRegex(text) {
    return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// Wraps every occurrence of any word from `query` in <mark> tags. Operates on
// already-HTML-escaped text so this only ever inserts safe, known markup.
function highlightMatches(escapedText, query) {
    const words = [...new Set(query.toLowerCase().split(/\s+/).filter(Boolean))];
    if (words.length === 0) return escapedText;

    const pattern = new RegExp(`(${words.map(escapeRegex).join("|")})`, "gi");
    return escapedText.replace(pattern, "<mark>$1</mark>");
}

function renderResults(results, query) {
    resultsEl.innerHTML = "";

    if (results.length === 0) {
        const empty = document.createElement("p");
        empty.className = "empty-state";
        empty.textContent = "No notes match that search.";
        resultsEl.appendChild(empty);
        return;
    }

    for (const r of results) {
        const card = document.createElement("div");
        card.className = "result-card";

        const safeText = highlightMatches(escapeHtml(r.raw_text), query);
        const scorePercent = Math.max(0, Math.min(1, r.score)) * 100;

        card.innerHTML = `
            <div class="result-body">
                <div class="result-meta">
                    <span class="result-path">${escapeHtml(r.file_path)}</span>
                    <span class="score-meter" title="Blended relevance score">
                        <span class="score-meter-track">
                            <span class="score-meter-fill" style="width: ${scorePercent}%;"></span>
                        </span>
                        <span class="score-meter-value">${r.score}</span>
                    </span>
                </div>
                <div class="result-text">${safeText}</div>
            </div>
        `;
        resultsEl.appendChild(card);
    }
}

function runSearch(query) {
    if (!query) {
        resultsEl.innerHTML = "";
        return;
    }
    resultsEl.classList.add("is-loading");
    fetch(`/api/search?q=${encodeURIComponent(query)}`)
        .then((res) => res.json())
        .then((results) => renderResults(results, query))
        .catch((err) => {
            resultsEl.innerHTML = `<p class="empty-state">Search failed: ${escapeHtml(String(err))}</p>`;
        })
        .finally(() => resultsEl.classList.remove("is-loading"));
}

searchBox.addEventListener("input", () => {
    clearTimeout(debounceTimer);
    const query = searchBox.value.trim();
    debounceTimer = setTimeout(() => runSearch(query), 300);
});

refreshBtn.addEventListener("click", () => {
    refreshBtn.disabled = true;
    refreshBtn.textContent = "Refreshing index...";
    refreshStatusEl.textContent = "";

    fetch("/api/refresh", { method: "POST" })
        .then((res) => res.json())
        .then((stats) => {
            refreshBtn.textContent = "Refresh index";
            refreshBtn.disabled = false;

            const parts = [];
            if (stats.files_indexed > 0) {
                parts.push(`Indexed ${stats.files_indexed} file(s), ${stats.chunks_indexed} chunk(s).`);
            } else {
                parts.push("Index already up to date.");
            }
            if (stats.unsupported_files && stats.unsupported_files.length > 0) {
                parts.push(`Skipped ${stats.unsupported_files.length} unsupported file(s) (not .md).`);
            }
            refreshStatusEl.textContent = parts.join(" ");
        })
        .catch((err) => {
            refreshBtn.textContent = "Refresh index";
            refreshBtn.disabled = false;
            refreshStatusEl.textContent = `Refresh failed: ${err}`;
        });
});
