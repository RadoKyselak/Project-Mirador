
const BACKEND_URL = "https://stelthar-api.vercel.app/verify";

const LOADING_GIF_URL = "https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExeWNjeHhsZHE2b2s0djI5dzRwYXZwOXhuanhob2ljN29sM3Z4dmJkeSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o85xCVo1diTHyIoPC/giphy.gif";

document.addEventListener("DOMContentLoaded", () => {
    const claimTextEl = document.getElementById("claim-text");
    const verifyBtn = document.getElementById("verifyBtn");
    const loadingIndicator = document.getElementById("loading-indicator");
    const resultsArea = document.getElementById("results-area");
    const verdictTextEl = document.getElementById("verdict-text");
    const confidenceTextEl = document.getElementById("confidence-text");
    const reliabilityEl = document.getElementById("reliability");
    const densityEl = document.getElementById("density");
    const alignmentEl = document.getElementById("alignment");
    const summaryTextEl = document.getElementById("summary-text");
    const evidenceListEl = document.getElementById("evidence-list");
    const errorMessageEl = document.getElementById("error-message");

    const loadingImg = loadingIndicator.querySelector('img');
    if (loadingImg) {
        loadingImg.src = LOADING_GIF_URL;
    }

    const clearResults = () => {
        resultsArea.style.display = 'none';
        errorMessageEl.textContent = '';
        verdictTextEl.textContent = '';
        verdictTextEl.className = '';
        confidenceTextEl.textContent = '';
        reliabilityEl.textContent = '';
        densityEl.textContent = '';
        alignmentEl.textContent = '';
        summaryTextEl.textContent = '';
        evidenceListEl.innerHTML = '';
    };

    chrome.storage.local.get(["stelthar_last_claim"], (data) => {
        const claim = data.stelthar_last_claim;
        if (claim) {
            claimTextEl.textContent = `"${claim}"`
        } else {
            claimTextEl.textContent = "No claim detected. Highlight text on a page and right-click or use the popup button.";
            verifyBtn.disabled = true;
        }
    });

    verifyBtn.onclick = async () => {
        clearResults();
        loadingIndicator.style.display = 'block';
        verifyBtn.disabled = true;
        errorMessageEl.textContent = '';

        chrome.storage.local.get(["stelthar_last_claim"], async (data) => {
            const claim = data.stelthar_last_claim;
            if (!claim) {
                errorMessageEl.textContent = "No claim found. Highlight text first.";
                loadingIndicator.style.display = 'none';
                return;
            }

            try {
                const resp = await fetch(BACKEND_URL, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ claim }),
                });

                if (!resp.ok) {
                    throw new Error(`Backend error: ${resp.status} ${resp.statusText}`);
                }

                const j = await resp.json();
                console.log("API Response:", j);

                verdictTextEl.textContent = `VERDICT: ${j.verdict || 'N/A'}`;
                verdictTextEl.className = (j.verdict || '').toLowerCase();

                confidenceTextEl.textContent = `CONFIDENCE: ${j.confidence?.toFixed(2) || 'N/A'} (${j.confidence_tier || 'N/A'})`;

                reliabilityEl.textContent = `- Source Reliability: ${j.confidence_breakdown?.source_reliability?.toFixed(2) || 'N/A'}`;
                densityEl.textContent = `- Evidence Density: ${j.confidence_breakdown?.evidence_density?.toFixed(2) || 'N/A'}`;
                alignmentEl.textContent = `- Semantic Alignment: ${j.confidence_breakdown?.semantic_alignment?.toFixed(2) || 'N/A'}`;

                summaryTextEl.textContent = j.summary || 'No summary provided.';

                evidenceListEl.innerHTML = '';
                if (j.evidence_links && j.evidence_links.length > 0) {
                    j.evidence_links.forEach(link => {
                        const li = document.createElement('li');
                        const findingText = link.finding || 'Link';
                        li.textContent = `• ${findingText}`;

                        if (link.source_url) {
                            const a = document.createElement('a');
                            a.href = link.source_url;
                            a.target = "_blank";
                            a.textContent = link.source_url;
                            li.appendChild(a);
                        }
                        evidenceListEl.appendChild(li);
                    });
                } else {
                    evidenceListEl.innerHTML = '<li>No specific evidence links provided.</li>';
                }

                resultsArea.style.display = 'block';

            } catch (e) {
                console.error("Verification error:", e);
                errorMessageEl.textContent = `Error: ${e.message}`;
            } finally {
                loadingIndicator.style.display = 'none';
                verifyBtn.disabled = false;
            }
        });
    };
});
