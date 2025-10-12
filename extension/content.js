document.addEventListener("mouseup", async (e) => {
  const selection = window.getSelection().toString().trim();
  if (!selection) return;
  const existing = document.getElementById("stelthar_verify_prompt");
  if (existing) existing.remove();
  const prompt = document.createElement("div");
  prompt.id = "stelthar_verify_prompt";
  prompt.style.position = "absolute";
  prompt.style.zIndex = 2147483647;
  prompt.style.background = "#111";
  prompt.style.color = "#fff";
  prompt.style.padding = "6px 8px";
  prompt.style.borderRadius = "6px";
  prompt.style.fontSize = "12px";
  prompt.style.cursor = "pointer";
  prompt.textContent = "Verify with Stelthar";
  document.body.appendChild(prompt);
  const rect = window.getSelection().getRangeAt(0).getBoundingClientRect();
  prompt.style.top = (window.scrollY + rect.top - 30) + "px";
  prompt.style.left = (window.scrollX + rect.left) + "px";
  const cleanup = () => {
    const el = document.getElementById("stelthar_verify_prompt");
    if (el) el.remove();
  };
  prompt.onclick = () => {
    cleanup();
    chrome.runtime.sendMessage({type: "verify", claim: selection});
  };
  setTimeout(cleanup, 4000);
});

