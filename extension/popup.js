const BACKEND_URL = "https://stelthar-api.vercel.app/verify";

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
        verifyBtn.disabled = true;
        loadingIndicator.style.display = 'flex';
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

                const jsonResp = await resp.json();
                console.log("Received response:", jsonResp);

                loadingIndicator.style.display = 'none';
                resultsArea.style.display = 'block';
                verifyBtn.disabled = false;

                verdictTextEl.textContent = `VERDICT: ${jsonResp.verdict.toUpperCase()}`;
                verdictTextEl.className = jsonResp.verdict.toLowerCase();
                confidenceTextEl.textContent = `CONFIDENCE: ${jsonResp.confidence}`;
                summaryTextEl.textContent = jsonResp.summary;

                evidenceLinksListEl.innerHTML = '';
                if (jsonResp.evidence_links && jsonResp.evidence_links.length > 0) {
                    jsonResp.evidence_links.forEach(link => {
                    const li = document.createElement('li');
                    const a = document.createElement('a');
                    a.href = link.source_url;
                    a.textContent = link.finding;
                    a.target = '_blank';
                    li.appendChild(a);
                    evidenceLinksListEl.appendChild(li);
                });
                } else {
                    evidenceLinksListEl.innerHTML = '<li>No evidence links provided.</li>';
                }

                sourcesListEl.innerHTML = '';
                if (jsonResp.sources && jsonResp.sources.length > 0) {
                    sourcesDropdown.style.display = 'block';
                    jsonResp.sources.forEach(source => {
                        const li = document.createElement('li');

                        const titleSpan = document.createElement('span');
                        titleSpan.className = 'source-title';
                        titleSpan.textContent = source.title;
                        li.appendChild(titleSpan);

                        if (source.snippet) {
                            const snippetSpan = document.createElement('span');
                            snippetSpan.className = 'source-snippet';
                            snippetSpan.textContent = `"${source.snippet}"`;
                            li.appendChild(snippetSpan);
                        }

                        const a = document.createElement('a');
                        a.href = source.url;
                        a.textContent = source.url;
                        a.target = '_blank';
                        li.appendChild(a);

                        sourcesListEl.appendChild(li);
                    });
                }

            } catch (error) {
                console.error("Verification failed:", error);
                loadingIndicator.style.display = 'none';
                errorMessageEl.textContent = `Error: ${error.message}`;
                errorMessageEl.style.display = 'block';
                verifyBtn.disabled = false;
            }
        });
    };
});


