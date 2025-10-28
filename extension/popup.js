const BACKEND_URL = "https://stelthar-api.vercel.app/verify";
const LOADING_GIF_URL = "https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExeWNjeHhsZHE2b2s0djI5dzRwYXZwOXhuanhob2ljN29sM3Z4dmJkeSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o85xCVo1diTHyIoPC/giphy.gif"; // Your chosen GIF

document.addEventListener("DOMContentLoaded", () => {

    const claimTextEl = document.getElementById("claim-text");
    const verifyBtn = document.getElementById("verifyBtn");
    const loadingIndicator = document.getElementById("loading-indicator");
    const resultsArea = document.getElementById("results-area");
    const verdictTextEl = document.getElementById("verdict-text");
    const confidenceTextEl = document.getElementById("confidence-text");
    const summaryTextEl = document.getElementById("summary-text");
    const evidenceLinksListEl = document.getElementById("evidence-links-list");
    const sourcesDropdown = document.getElementById("sources-dropdown");
    const sourcesListEl = document.getElementById("sources-list");
    const errorMessageEl = document.getElementById("error-message");

    const loadingImg = loadingIndicator.querySelector('img');
    if (loadingImg) {
        loadingImg.src = LOADING_GIF_URL;
    }

    const clearResults = () => {
        resultsArea.style.display = 'none';
        sourcesDropdown.style.display = 'none';
        sourcesDropdown.open = false;
        errorMessageEl.style.display = 'none';
        errorMessageEl.textContent = '';
        verdictTextEl.textContent = 'VERDICT: ...';
        verdictTextEl.className = '';
        confidenceTextEl.textContent = 'CONFIDENCE: ...';
        summaryTextEl.textContent = 'Loading summary...';
        evidenceLinksListEl.innerHTML = '';
        sourcesListEl.innerHTML = '';
    };

    chrome.storage.local.get(["stelthar_last_claim"], (data) => {
        const claim = data.stelthar_last_claim;
        console.log("Popup loaded. Claim from storage:", claim);
        if (claim) {
            claimTextEl.textContent = `"${claim}"`;
            verifyBtn.disabled = false;
        } else {
            claimTextEl.textContent = "No claim detected. Highlight text on a page first, then click the 'Verify' button that appears.";
            verifyBtn.disabled = true;
        }
    });

    verifyBtn.onclick = async () => {
        clearResults();
        loadingIndicator.style.display = 'block';
        verifyBtn.disabled = true;

        chrome.storage.local.get(["stelthar_last_claim"], async (data) => {
            const claim = data.stelthar_last_claim;
            if (!claim) {
                errorMessageEl.textContent = "No claim found in storage. Highlight text first.";
                errorMessageEl.style.display = 'block';
                loadingIndicator.style.display = 'none';
                return;
            }

            console.log("Verifying claim:", claim);

            try {
                const resp = await fetch(BACKEND_URL, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ claim }),
                });

                if (!resp.ok) {
                    let errorDetail = resp.statusText;
                    try {
                        const errorJson = await resp.json();
                        errorDetail = errorJson.detail || errorDetail;
                    } catch (e) {}
                    throw new Error(`Backend error: ${resp.status} ${errorDetail}`);
                }

                const j = await resp.json();
                console.log("API Response Received:", j);

                verdictTextEl.textContent = `VERDICT: ${j.verdict || 'N/A'}`;
                verdictTextEl.className = '';
                if (j.verdict) {
                    verdictTextEl.classList.add((j.verdict).toLowerCase());
                }

                confidenceTextEl.textContent = `CONFIDENCE: ${j.confidence?.toFixed(2) || 'N/A'} (${j.confidence_tier || 'N/A'})`;

                summaryTextEl.textContent = j.summary || 'No summary provided.';

                evidenceLinksListEl.innerHTML = '';
                if (j.evidence_links && j.evidence_links.length > 0) {
                    j.evidence_links.forEach(link => {
                        const li = document.createElement('li');
                        const findingSpan = document.createElement('span');
                        findingSpan.className = 'finding';
                        findingSpan.textContent = `• ${link.finding || 'Link'}`;
                        li.appendChild(findingSpan);

                        if (link.source_url) {
                            const a = document.createElement('a');
                            a.href = link.source_url;
                            a.target = "_blank";
                            a.textContent = link.source_url;
                            li.appendChild(a);
                        }
                        evidenceLinksListEl.appendChild(li);
                    });
                } else {
                    evidenceLinksListEl.innerHTML = '<li>No specific evidence links found.</li>';
                }

                sourcesListEl.innerHTML = '';
                if (j.sources && j.sources.length > 0) {
                    j.sources.forEach(source => {
                        const li = document.createElement('li');

                        const titleSpan = document.createElement('span');
                        titleSpan.className = 'source-title';
                        titleSpan.textContent = source.title || 'Untitled Source';
                        li.appendChild(titleSpan);

                        if (source.snippet) {
                            const snippetSpan = document.createElement('span');
                            snippetSpan.className = 'source-snippet';
                            snippetSpan.textContent = source.snippet.substring(0, 150) + (source.snippet.length > 150 ? '...' : '');
                            li.appendChild(snippetSpan);
                        }

                        if (source.url) {
                            const a = document.createElement('a');
                            a.href = source.url;
                            a.target = "_blank";
                            a.textContent = "View Source";
                            li.appendChild(a);
                        }
                        sourcesListEl.appendChild(li);
                    });
                    sourcesDropdown.style.display = 'block';
                } else {
                    sourcesDropdown.style.display = 'none';
                }

                resultsArea.style.display = 'block';

            } catch (e) {
                console.error("Verification process error:", e);
                errorMessageEl.textContent = `Error during verification: ${e.message}`;
                errorMessageEl.style.display = 'block';
            } finally {
                loadingIndicator.style.display = 'none';
                verifyBtn.disabled = false;
            }
        });
    };
}); // End DOMContentLoaded
