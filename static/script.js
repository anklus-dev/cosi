// script.js
// Handles the search bar's real-time autocompletion (debounced fetch to /api/search)
// and renders ranked paragraph cards into #results. Also wires up #refresh-btn
// to POST /api/refresh (the "Prompted Refresh" manual re-index action).

const searchBox = document.getElementById("search-box");
const resultsEl = document.getElementById("results");
const refreshBtn = document.getElementById("refresh-btn");

let debounceTimer = null;

function renderResults(results) {
    resultsEl.innerHTML = "";

    if (results.length === 0) {
        resultsEl.innerHTML = "<p>No matches yet.</p>";
        return;
    }

    for (const r of results) {
        const card = document.createElement("div");
        card.className = "result-card";
        card.innerHTML = `
            <div class="result-score">score: ${r.score}</div>
            <div class="result-path">${r.file_path}</div>
            <div class="result-text">${r.raw_text}</div>
        `;
        resultsEl.appendChild(card);
    }
}

function runSearch(query) {
    if (!query) {
        resultsEl.innerHTML = "";
        return;
    }
    fetch(`/api/search?q=${encodeURIComponent(query)}`)
        .then((res) => res.json())
        .then(renderResults)
        .catch((err) => {
            resultsEl.innerHTML = `<p>Search failed: ${err}</p>`;
        });
}

searchBox.addEventListener("input", () => {
    clearTimeout(debounceTimer);
    const query = searchBox.value.trim();
    debounceTimer = setTimeout(() => runSearch(query), 300);
});

refreshBtn.addEventListener("click", () => {
    refreshBtn.disabled = true;
    refreshBtn.textContent = "Refreshing...";
    fetch("/api/refresh", { method: "POST" })
        .then((res) => res.json())
        .then(() => {
            refreshBtn.textContent = "Refresh Index";
            refreshBtn.disabled = false;
        });
});
