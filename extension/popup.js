const BACKEND_URL = "https://stelthar-api.vercel.app/verify";

document.addEventListener("DOMContentLoaded", () => {
  const claimEl = document.getElementById("claim");
  const resultEl = document.getElementById("result");
  const btn = document.getElementById("verifyBtn");
  chrome.storage.local.get(["stelthar_last_claim"], (data) => {
    const c = data.stelthar_last_claim;
    if (c) {
      claimEl.textContent = c;
    } else {
      claimEl.textContent = "No claim detected. Highlight text and click Verify in the page.";
    }
  });

  btn.onclick = async () => {
    resultEl.innerHTML = "Verifying…";
    chrome.storage.local.get(["stelthar_last_claim"], async (data) => {
      const claim = data.stelthar_last_claim;
      if (!claim) {
        resultEl.innerHTML = "No claim found. Highlight text first.";
        return;
      }
      try {
        const resp = await fetch(BACKEND_URL, {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({claim})
        });
        const j = await resp.json();
        
        resultEl.innerHTML = `
          <div class="verdict">Verdict: ${j.verdict} (confidence: ${j.confidence})</div>
          <div style="margin-top:8px;"><strong>Normalized:</strong> ${j.claim_normalized}</div>
          <div style="margin-top:8px;"><strong>Summary:</strong> ${j.summary}</div>
          <div style="margin-top:8px;"><strong>Sources:</strong></div>
          <div>${(j.sources || []).map(s => `<div class="source"><a href="${s.url}" target="_blank">${s.title}</a><div style="font-size:11px">${s.snippet.substring(0,200)}...</div></div>`).join("")}</div>
        `;
      } catch (e) {
        resultEl.innerHTML = "Error contacting backend: " + e.message;
      }
    });
  };
});


